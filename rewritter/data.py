from torch.utils.data import Dataset
from collections import defaultdict
from itertools import chain
import numpy as np
import jieba
import re


class RewritterDataset(Dataset):
    def __init__(self, file_path, tokenize, max_src_len=100, max_tgt_len=50, min_tf=2):
        super().__init__()
        self.tokenize = tokenize
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.min_tf = min_tf
        self._load_data(file_path)
        self._tokenize()
        self._build_vocabulary()
        self._convert_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sentences = self.data[index]
        src_seq = list(chain.from_iterable(sentences[:-1]))
        src_seq = src_seq[-self.max_src_len:] + [0] * (self.max_src_len - len(src_seq))
        turns = []
        segment_type = []
        for i, sentence in enumerate(sentences[:-1]):
            turns.extend([i] * len(sentence))
            segment_type.extend([0] * len(sentences))
        turns.extend([2] * len(sentences[-1]))
        segment_type.extend([1] * len(sentences[-1]))
        src_turns = turns[-self.max_src_len:] + [2] * (self.max_src_len - len(turns))
        src_turns = np.array(src_turns, dtype=np.int64)
        segment_type = segment_type[-self.max_src_len:] + [1] * (self.max_src_len - len(segment_type))
        segment_type = np.array(segment_type, dtype=np.int64)
        tgt_seq = [self.vocab["<SOS>"]] + sentences[-1]
        tgt_seq = tgt_seq[:self.max_tgt_len] + [0] * (self.max_tgt_len - len(tgt_seq))
        tgt_turns = [3] * self.max_tgt_len
        tgt_turns = np.array(tgt_turns, dtype=np.int64)
        words = self._get_distinct_words(src_seq, tgt_seq)
        transform_matrx = self._compute_transform_matrix(words, src_seq)
        labels = self._get_labels(words, tgt_seq)
        src_seq = np.array(src_seq, dtype=np.int64)
        tgt_seq = np.array(tgt_seq, dtype=np.int64)
        return src_seq, tgt_seq, src_turns, tgt_turns, segment_type, transform_matrx, labels

    def _get_distinct_words(self, src_seq, tgt_seq):
        words = []
        for token in src_seq + tgt_seq[1:]:
            if token != 0 and token not in words:
                words.append(token)
        return words

    def _get_labels(self, words, tgt_seq):
        labels = []
        for token in tgt_seq[1:]:
            if token in words:
                labels.append(words.index(token))
            else:
                labels.append(-1)
        labels.append(-1)
        return np.array(labels, dtype=np.int64)

    def _compute_transform_matrix(self, words, src_seq):
        src_seq = src_seq.copy()
        matrix = [[0] * len(src_seq) for _ in range(len(src_seq))]
        for i, word in enumerate(words):
            if word in src_seq:
                idx = src_seq.index(word)
                while idx != -1:
                    matrix[idx][i] = 1
                    src_seq[idx] = -1
                    idx = src_seq.index(word) if word in src_seq else -1
        return np.array(matrix, dtype=np.float32)

    def _load_data(self, file_path):
        with open(file_path, encoding="utf-8") as fi:
            self.data = [line.strip().split("\t\t") for line in fi]

    def _tokenize(self):
        tokenized_data = []
        for sentences in self.data:
            tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
            for sentence in tokenized_sentences:
                sentence.append("<EOS>")  # end of sentence
            tokenized_data.append(tokenized_sentences)
        self.data = tokenized_data

    def _build_vocabulary(self):
        counts = defaultdict(int)
        for sentences in self.data:
            for token in chain.from_iterable(sentences):
                counts[token] += 1
        words = [word for word, count in counts.items() if count >= self.min_tf]
        words.append("<SOS>")  # start of sentence
        words.append("<UNK>")
        self.vocab = {word: i + 1 for i, word in enumerate(words)}  # 0 reserved for padding

    def _convert_data(self):
        converted_data = []
        for sentences in self.data:
            converted_data.append(
                [[self.vocab.get(token, self.vocab["<UNK>"]) for token in sentence]
                 for sentence in sentences]
            )
        self.data = converted_data


p = re.compile("[a-zA-Z0-9]+")


def tokenize(utterance):
    tokens = jieba.lcut(utterance)
    flat_tokens = []
    for token in tokens:
        if p.match(token):
            flat_tokens.append(token)
        else:
            flat_tokens.extend(token)
    return flat_tokens



if __name__ == "__main__":
    data = RewritterDataset("data/corpus.txt", lambda x: list(x))
    for i in range(len(data)):
        src_seq = data[i][0]
        s1 = (src_seq != 0).sum()
        s2 = data[i][-2].sum()
        if s1 != s2:
            print(s1, s2)
    for i in range(len(data)):
        src, tgt = data[i][0], data[i][1]
        if 3619 in src or 3619 in tgt:
            print(src)
            print(tgt)
            break
