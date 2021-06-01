from itertools import chain
from math import log
import torch
import copy
import heapq


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def __len__(self):
        return len(self._queue)

    def push(self, priority, item):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def empty(self):
        return len(self._queue) == 0
    
    def clear(self):
        self._queue.clear()
        self._index = 0


class BeamSearchNode:
    def __init__(self, vocab, words, max_len):
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.token_idxes = [vocab["<SOS>"]]
        self.words = words
        self.tokens = ["<SOS>"]
        self.logp = 0
        self.score = 0
        self.max_len = max_len

    def append(self, label_idx, logp):
        if self.tokens[-1] == "<EOS>" or len(self.tokens) == self.max_len:
            return
        token_idx = self.words[label_idx]
        self.token_idxes.append(token_idx)
        self.tokens.append(self.inv_vocab.get(self.words[label_idx]))
        self.logp += logp
        self.score = self.logp / (len(self.token_idxes) - 1 + 1e-6)

    def is_endnode(self):
        return self.tokens[-1] == "<EOS>" or len(self.tokens) == self.max_len

    def get_utterance(self):
        return "".join([token for token in self.tokens if token not in ["<SOS>", "<EOS>", "<UNK>"]])


class BeamSearchDecoder:
    def __init__(self, model, tokenize, beam_size, vocab, max_src_len, max_len, history_size=2, device="cpu"):
        self.model = model
        self.tokenize = tokenize
        self.beam_size = beam_size
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_len = max_len
        self.history_size = history_size
        self.device = device

    def inference(self, history_utterances, current_utterance):
        src_seqs, src_turns, segment_type, transform_matrix, words = self._convert_src_input(history_utterances, current_utterance)
        src = self.model.encode(src_seqs, src_turns)  # 1 * s * e
        src_mask = (src_seqs != 0).unsqueeze(1)
        nodes = PriorityQueue()
        nodes.push(0, BeamSearchNode(self.vocab, words, self.max_len))
        endnodes = []
        while not nodes.empty():
            selected_nodes = self._select_nodes(nodes, endnodes)
            if len(endnodes) == self.beam_size or not selected_nodes:
                break
            nodes.clear()
            src_ = src.repeat(len(selected_nodes), 1, 1) # beam_size * s * e
            src_mask_ = src_mask.repeat(len(selected_nodes), 1, 1)
            segment_type_ = segment_type.repeat(len(selected_nodes), 1)
            transform_matrix_ = transform_matrix.repeat(len(selected_nodes), 1, 1)
        
            tgt_seqs, tgt_turns = self._convert_tgt_input(selected_nodes)
            logps = self.model.decode(src_, src_mask_, tgt_seqs, tgt_turns, segment_type_, transform_matrix_)
            self._select_topk(nodes, selected_nodes, logps)

        utterances_with_socres = [(node.get_utterance(), node.score) for node in endnodes]
        return utterances_with_socres
    
    def _select_nodes(self, nodes, endnodes):
        selected_nodes = []
        while len(selected_nodes) < self.beam_size and not nodes.empty():
            node = nodes.pop()
            if node.is_endnode():
                endnodes.append(node)
                if len(endnodes) == self.beam_size:
                    break
                continue
            selected_nodes.append(node)
        return selected_nodes

    def _select_topk(self, nodes, selected_nodes, logps):
        logps = logps[:,-1]
        top_logps, top_idxes = torch.topk(logps, self.beam_size)
        top_logps = top_logps.cpu().detach().numpy()
        top_idxes = top_idxes.cpu().detach().numpy()

        for i in range(len(selected_nodes)):
            node = selected_nodes[i]
            for logp, idx in zip(top_logps[i], top_idxes[i]): 
                new_node = copy.deepcopy(node)
                new_node.append(idx, logp)
                nodes.push(new_node.score, new_node)

    def _convert_src_input(self, history_utterances, current_utterance):
        history_tokens = [self.tokenize(utterance) for utterance in history_utterances]
        for tokens in history_tokens:
            tokens.append("<EOS>")
        src_turns = []
        segment_type = []
        for i, tokens in enumerate(history_tokens):
            src_turns.extend([i] * len(tokens))
            segment_type.extend([0] * len(tokens))
        current_tokens = self.tokenize(current_utterance)
        current_tokens.append("<EOS>")
        src_turns.extend([len(history_tokens)] * len(current_tokens))
        src_turns = src_turns[-self.max_src_len:] + [len(history_tokens)] * (self.max_src_len - len(src_turns)) 
        segment_type.extend([1] * len(current_tokens))
        segment_type = segment_type[-self.max_src_len:] + [1] * (self.max_src_len - len(segment_type))
        token_idxes = []
        for token in chain.from_iterable(history_tokens):
            token_idxes.append(self.vocab.get(token, self.vocab["<UNK>"]))
        for token in current_tokens:
            token_idxes.append(self.vocab.get(token, self.vocab["<UNK>"]))
        token_idxes = token_idxes[-self.max_src_len:] + [0] * (self.max_src_len - len(token_idxes))
        words = self._get_distinct_words(token_idxes)
        transform_matrix = self._compute_transform_matrix(words, token_idxes)
        src_seqs = torch.tensor([token_idxes], dtype=torch.long, device=self.device)
        src_turns = torch.tensor([src_turns], dtype=torch.long, device=self.device)
        segment_type = torch.tensor([segment_type], dtype=torch.long, device=self.device)
        return src_seqs, src_turns, segment_type, transform_matrix, words
    
    def _convert_tgt_input(self, selected_nodes):
        token_idxes = [node.token_idxes for node in selected_nodes]
        tgt_seqs = torch.tensor(token_idxes, dtype=torch.long, device=self.device)
        tgt_turns = [[self.history_size+1] * len(token_idxes[0]) for _ in range(len(selected_nodes))]
        tgt_turns = torch.tensor(tgt_turns, dtype=torch.long, device=self.device)
        return tgt_seqs, tgt_turns
    
    def _get_distinct_words(self, src_seq):
        words = []
        for token in src_seq:
            if token != 0 and token not in words:
                words.append(token)
        return words
            
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
        return torch.tensor([matrix], dtype=torch.float, device=self.device)
