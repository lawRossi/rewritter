from torch.utils.data import DataLoader
from rewritter.dataset import RewritterDataset
from .new_model import LstmRewriterModel, TransformerRewriterModel, BertRewriterModel
from torch.optim import Adam
import json
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from itertools import chain


def train(bert=False):
    train_data = RewritterDataset("data/train.json", bert_dataset=bert)
    test_data = RewritterDataset("data/test.json", bert_dataset=bert)

    train_data_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=2)

    if not bert:
        with open("data/vocab.json", encoding="utf-8") as fi:
            vocab = json.load(fi)
        vocab_size = len(vocab) + 1

    with open("data/params.json", "w") as fo:
        json.dump({"max_ctx_len": train_data.max_context_len, "max_utr_len": train_data.max_utterance_len}, fo)
    class_weights = torch.tensor([0.25, 0.4, 0.35])
    # model = LstmRewriterModel(vocab_size, 100, 400, class_weights=class_weights)
    # model = TransformerRewriterModel(vocab_size, 100, class_weights=class_weights)
    model = BertRewriterModel("C:/code/models/chinese_base", class_weights, segment_type="unet")
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = Adam(model.parameters(), lr=1e-5)

    total_loss = 0
    device = "cpu"
    torch.save(model, "data/model.pt")
    epochs = 1
    for epoch in range(epochs):
        print(f"epoch{epoch}")
        model.train()
        for i, batch in enumerate(train_data_loader):
            if not bert:
                batch = {k: v.to(device) for k, v in batch.items()}
            contexts = batch["context"]
            utterances = batch["utterance"]
            labels = batch["labels"]
            if bert:
                contexts = {k: v.to(device) for k, v in contexts.items()}
                utterances = {k: v.to(device) for k, v in utterances.items()}
                labels = labels.to(device)
            loss = model(contexts, utterances, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(total_loss / 10)
                total_loss = 0
            break
        test(model, test_data_loader, device, bert)
        if epoch > 1:
            model_path = f"data/model{epoch-1}.pt"
            torch.save(model, model_path)


def test(model, data_loader, device, bert=False):
    pbar = tqdm(data_loader)
    all_labels = []
    all_preds = []
    model.eval()
    for i, batch in enumerate(pbar):
        if not bert:
            batch = {k: v.to(device) for k, v in batch.items()}
        contexts = batch["context"]
        utterances = batch["utterance"]
        if bert:
            contexts = {k: v.to(device) for k, v in contexts.items()}
            utterances = {k: v.to(device) for k, v in utterances.items()}
        labels = batch["labels"].cpu().numpy()
        logits = model(contexts, utterances)
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().detach().numpy()
        for label, pred in zip(chain.from_iterable(labels), chain.from_iterable(preds)):
            if label != -1:
                all_labels.append(label)
                all_preds.append(pred)
        if i == 1:
            break
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    train(True)
