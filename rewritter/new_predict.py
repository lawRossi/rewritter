import torch 
import numpy as np
from collections import Counter
from .preprocess import get_operations, translate
import os.path
import json
from . import bleu
from .import rouge


class ModelWrapper:
    def __init__(self, model_path, tokenize=None, device="cpu", bert_model=False):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        model_dir = os.path.dirname(model_path)
        if not bert_model:
            with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fi:
                self.vocab = json.load(fi)
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.tokenize = tokenize
        else:
            self.inv_vocab = {v: k for k, v in self.model.tokenizer.vocab.items()}
            self.tokenize = self.model.tokenizer

        with open(os.path.join(model_dir, "params.json"), encoding="utf-8") as fi:
            params = json.load(fi)
            self.max_ctx_len = params["max_ctx_len"]
            self.max_utr_len = params["max_utr_len"]
        self.device = device
        self.bert_model = bert_model

    def predict(self, contexts, utterances, method="original", appendix=""):
        if not self.bert_model:
            contexts_array, utterances_array, contexts_tensor, utterances_tensor, masks = self._convert_inputs(contexts, utterances, appendix)
        else:
            contexts_array, utterances_array, contexts_tensor, utterances_tensor, masks = self._convert_bert_inputs(contexts, utterances, appendix)
        logits = self.model(contexts_tensor, utterances_tensor)
        labels = torch.softmax(logits, dim=-1).argmax(dim=-1)
        matrixes = labels.reshape(len(contexts), self.max_ctx_len, self.max_utr_len)
        matrixes = matrixes.cpu().detach().numpy()
        predicted_texts = []
        for i, matrix in enumerate(matrixes):
            matrix = matrix * masks[i]
            if method == "original":
                operations = self._derive_operations_(contexts_array[i], matrix)
            else:
                operations = self._derive_operations(contexts_array[i], matrix)
            translations = translate(utterances_array[i], operations)
            translation_texts = []
            for translation in translations:
                tokens = [self.inv_vocab.get(idx) for idx in translation if idx in self.inv_vocab]
                translation_texts.append(self._tokens2text(tokens))
            predicted_texts.append(translation_texts)
        return predicted_texts

    def _convert_inputs(self, contexts, utterances, appendix):
        contexts_array, utterances_array, masks = self._text2array(contexts, utterances, appendix)
        contexts_tensor = self._array2tensor(contexts_array)
        utterances_tensor = self._array2tensor(utterances_array)
        return contexts_array, utterances_array, contexts_tensor, utterances_tensor, masks

    def _derive_operations(self, contexts, matrix):
        connect_matrix = np.where(matrix != 0, 1, 0)
        boxes = self._scan_twice(connect_matrix)
        for box in boxes:
            x1, x2 = box[0]
            y1, y2 = box[1]
            counts = Counter(matrix[y1:y2, x1:x2].flatten())
            if 0 in counts:
                del counts[0]
            label = counts.most_common(1)[0][0]
            matrix[y1:y2, x1:x2] = label
        operations = get_operations(contexts, matrix)
        return operations

    def _derive_operations_(self, contexts, matrix):
        connect_matrix = np.where(matrix == 1, 1, 0)
        boxes = self._scan_twice(connect_matrix)
        for box in boxes:
            x1, x2 = box[0]
            y1, y2 = box[1]
            matrix[y1:y2, x1:x2] = 1
        operations1 = get_operations(contexts, matrix)

        connect_matrix = np.where(matrix == 2, 1, 0)
        boxes = self._scan_twice(connect_matrix)
        for box in boxes:
            x1, x2 = box[0]
            y1, y2 = box[1]
            matrix[y1:y2, x1:x2] = 2
        operations2 = get_operations(contexts, matrix)
    
        return operations1 + operations2

    def _tokens2text(self, tokens):
        prev_is_not_chinese = False
        text = ""
        for token in tokens:
            if self._is_chinese(token):
                text += token
                prev_is_not_chinese = False
            else:
                if prev_is_not_chinese:
                    text += " " + token    
                else:
                    text += token
                prev_is_not_chinese = True
        text = text.replace("[SEP]", "")
        text = text.replace("[PAD]", "")
        text = text.strip()
        return text            

    def _is_chinese(self, token):
        return '\u4e00' <= token <= '\u9fa5'

    def _text2array(self, contexts, utterances, appendix):
        contexts_arrays = []
        utterance_arrays = []
        masks = []
        for context, utterance in zip(contexts, utterances):
            context_tokens = []
            for text in context:
                tokens = self.tokenize(text)
                tokens.append("[SEP]")
                context_tokens.extend(tokens)
            if appendix:
                context_tokens.extend(appendix.split(" "))
            context_idxes = [self.vocab.get(token, self.vocab["<UNK>"]) for token in context_tokens]
            context_idxes = context_idxes[-self.max_ctx_len:] + [0] * (self.max_ctx_len - len(context_idxes))
            utterance_tokens = self.tokenize(utterance)
            utterance_tokens.append("[SEP]")
            utterance_idxes = [self.vocab.get(token, self.vocab["<UNK>"]) for token in utterance_tokens]
            utterance_idxes = utterance_idxes[-self.max_utr_len:] + [0] * (self.max_utr_len - len(utterance_idxes))
            contexts_arrays.append(context_idxes)
            utterance_arrays.append(utterance_idxes)
            context_mask = np.array(context_idxes) != 0
            utterance_mask = np.array(utterance_idxes) != 0
            mask = np.outer(context_mask, utterance_mask)
            masks.append(mask)

        return contexts_arrays, utterance_arrays, masks

    def _array2tensor(self, array):
        return torch.tensor(array, dtype=torch.long, device=self.device)
    
    def _convert_bert_inputs(self, contexts, utterances, appendix):
        new_contexts = []
        for context in contexts:
            new_contexts.append("[SEP]".join(context))
        context_data = self.tokenize(new_contexts, padding="max_length", max_length=self.max_ctx_len, truncation=True)
        utterance_data = self.tokenize(utterances, padding="max_length", max_length=self.max_utr_len+1, truncation=True)
        for k in utterance_data:
            for i in range(len(contexts)):
                utterance_data[k][i] = utterance_data[k][i][1:]  # drop [cls]
        contexts_array = context_data["input_ids"]
        utterances_array = utterance_data["input_ids"]
        masks = []
        for context_idxes, utterance_idxes in zip(contexts_array, utterances_array):
            mask = np.outer((np.array(context_idxes) != 0), (np.array(utterance_idxes) != 0))
            masks.append(mask)
        context_tensors = {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in context_data.items()}
        utterance_tensors = {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in utterance_data.items()}
        return contexts_array, utterances_array, context_tensors, utterance_tensors, masks

    def _scan_twice(self, matrix):
        label_num = 1
        label_equations = {}
        height, width = matrix.shape
        for i in range(height):
            for j in range(width):
                if matrix[i, j] == 0:
                    continue
                if j != 0:
                    left_val = matrix[i, j - 1]
                else:
                    left_val = 0
                if i != 0:
                    top_val = matrix[i - 1, j]
                else:
                    top_val = 0
                if i != 0 and j != 0:
                    left_top_val = matrix[i - 1, j - 1]
                else:
                    left_top_val = 0
                if any([left_val > 0, top_val > 0, left_top_val > 0]):
                    neighbour_labels = [v for v in [left_val, top_val,
                                                    left_top_val] if v > 0]
                    min_label = min(neighbour_labels)
                    matrix[i, j] = min_label
                    set_min_label = min([label_equations[label] for label in
                                         neighbour_labels])
                    for label in neighbour_labels:
                        label_equations[label] = min(set_min_label, min_label)
                    if set_min_label > min_label:
                        for key, value in label_equations:
                            if value == set_min_label:
                                label_equations[key] = min_label
                else:
                    new_label = label_num
                    matrix[i, j] = new_label
                    label_equations[new_label] = new_label
                    label_num += 1
        for i in range(height):
            for j in range(width):
                if matrix[i, j] == 0:
                    continue
                label = matrix[i, j]
                normalized_label = label_equations[label]
                matrix[i, j] = normalized_label
        groups = list(set(label_equations.values()))
        ret_boxes = []
        for group_label in groups:
            points = np.argwhere(matrix == group_label)
            points_y = points[:, (0)]
            points_x = points[:, (1)]
            min_width = np.amin(points_x)
            max_width = np.amax(points_x) + 1
            min_height = np.amin(points_y)
            max_height = np.amax(points_y) + 1
            ret_boxes.append([[min_width, max_width], [min_height, max_height]])
        return ret_boxes


def evaluate_model(model_path, tokenize, test_file, result_file, batch_size=32, 
        device="cpu", bert_model=False):
    model = ModelWrapper(model_path, tokenize, device=device, bert_model=bert_model)
    with open(test_file, encoding="utf-8") as fi:
        samples = []
        for line in fi:
            sample = json.loads(line)
            raw_sample = sample["raw_sample"]
            context = raw_sample[:-2]
            utterance = raw_sample[-2]
            reference = raw_sample[-1]
            samples.append((context, utterance, reference))

    with open(result_file, "w", encoding="utf-8") as fo:
        for i in range(0, len(samples), batch_size):
            batch_contexts = [sample[0] for sample in samples[i:i+batch_size]]
            batch_utterances = [sample[1] for sample in samples[i:i+batch_size]]
            batch_references = [sample[2] for sample in samples[i:i+batch_size]]
            texts = model.predict(batch_contexts, batch_utterances)
            for text, reference in zip(texts, batch_references):
                fo.write(text[0] + "\t" + reference + "\n")
    evaluate(result_file, tokenize)    


def evaluate(result_file, tokenize):
    em = 0
    n = 0
    predict_tokens = []
    reference_tokens = []
    predict_texts = []
    reference_texts = []
    with open(result_file, encoding="utf-8") as fi:
        for line in fi:
            n += 1
            text, reference = line.rstrip().split("\t")
            if text == reference:
                em += 1
            pred_tokens = tokenize(text)
            ref_tokens = tokenize(reference)
            predict_tokens.append(pred_tokens)
            reference_tokens.append([ref_tokens])
            predict_texts.append(" ".join(pred_tokens))
            reference_texts.append(" ".join(ref_tokens))
    result = bleu.compute_bleu(reference_tokens, predict_tokens)
    bleus = result[1]
    print([f"bleu{i+1}: {bleus[i]}" for i in range(4)])
    result = rouge.rouge(predict_texts, reference_texts)
    print([f"{key}: {result[key]}" for key in ["rouge_1/f_score", "rouge_2/f_score", "rouge_l/f_score"]])
    print(f"em: {em / n}")
