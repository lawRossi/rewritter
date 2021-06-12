import torch 
import numpy as np
from collections import Counter
from .preprocess import get_operations, translate


class ModelWrapper:
    def __init__(self, model_path, vocab, tokenize, max_ctx_len, max_utr_len, device="cpu"):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.tokenize = tokenize
        self.max_ctx_len = max_ctx_len
        self.max_utr_len = max_utr_len
        self.device = device

    def predict(self, contexts, utterances):
        contexts_array, utterances_array, masks = self._text2array(contexts, utterances)
        contexts_tensor = self._array2tensor(contexts_array)
        utterances_tensor = self._array2tensor(utterances_array)
        logits = self.model(contexts_tensor, utterances_tensor)
        labels = torch.softmax(logits, dim=-1).argmax(dim=-1)
        matrixes = labels.reshape(contexts_tensor.shape[0], self.max_ctx_len, self.max_utr_len)
        matrixes = matrixes.cpu().detach().numpy()
        target_texts = []
        for i, matrix in enumerate(matrixes):
            matrix = matrix * masks[i]
            operations = self._derive_operations_(contexts_array[i], matrix)
            target = translate(utterances_array[i], operations)
            tokens = [self.inv_vocab.get(idx) for idx in target if idx in self.inv_vocab]
            target_texts.append(self._tokens2text(tokens))
        return target_texts
    
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
        text = text.replace("<SEP>", "")
        return text            

    def _is_chinese(self, token):
        return '\u4e00' <= token <= '\u9fa5'

    def _text2array(self, contexts, utterances):
        contexts_arrays = []
        utterance_arrays = []
        masks = []
        for context, utterance in zip(contexts, utterances):
            context_tokens = []
            for text in context:
                tokens = self.tokenize(text)
                tokens.append("<SEP>")
            context_tokens.extend(tokens)
            context_idxes = [self.vocab.get(token, self.vocab["<UNK>"]) for token in context_tokens]
            context_idxes = context_idxes[-self.max_ctx_len:] + [0] * (self.max_ctx_len - len(context_idxes))
            utterance_tokens = self.tokenize(utterance)
            utterance_tokens.append("<SEP>")
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
