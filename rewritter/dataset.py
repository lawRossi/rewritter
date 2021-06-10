from torch.utils.data import Dataset
import json
import numpy as np


class RewritterDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.max_context_len = None
        self.max_utterance_len = None
        self.samples = []
        with open(file_path, encoding="utf-8") as fi:
            for line in fi:
                sample = json.loads(line)
                if self.max_context_len is None:
                    self.max_context_len = len(sample["context"])
                if self.max_utterance_len is None:
                    self.max_utterance_len = len(sample["utterance"])
                del(sample["raw_sample"])
                sample["context"] = np.array(sample["context"], dtype=np.int64)
                sample["utterance"] = np.array(sample["utterance"], dtype=np.int64)
                sample["labels"] = np.array(sample["labels"], dtype=np.int64)
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


if __name__ == "__main__":
    data = RewritterDataset("../data/train.json")
    print(len(data))
    sample = data[2]
    print(sample["context"].shape)
    print(sample["utterance"].shape)
    print(sample["labels"].shape)

    data = RewritterDataset("../data/test.json")
    print(len(data))
    sample = data[2]
    print(sample["context"].shape)
    print(sample["utterance"].shape)
    print(sample["labels"].shape)
