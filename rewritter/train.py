from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from .model import ReWritterModel
from .data import RewritterDataset
from torch.optim import Adam


def train():
    data = RewritterDataset("data/corpus.txt", lambda x: list(x))
    data_loader = DataLoader(data, batch_size=4, shuffle=True)
    loss_func = CrossEntropyLoss(ignore_index=-1)
    model = ReWritterModel(len(data.vocab)+1 , 50, 5, 2, data.max_src_len)

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = Adam(model.parameters(), lr=1e-4)

    total_loss = 0
    for i, batch in enumerate(data_loader):
        src_seq, tgt_seq, src_turns, tgt_turns, segment_type, transform_matrx, labels = batch
        scores = model(src_seq, tgt_seq, src_turns, tgt_turns, segment_type, transform_matrx)
        loss = loss_func(scores.view(-1, scores.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        model.zero_grad()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(total_loss / 10)
            total_loss = 0


if __name__ == "__main__":
    train()
