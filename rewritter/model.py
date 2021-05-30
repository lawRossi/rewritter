import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).to(x.device)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dims, max_seq_len=100):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        init_range = 0.5 / emb_dims
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.position_embedding = PositionalEmbedding(emb_dims, max_seq_len)
        self.turn_embedding = nn.Embedding(4, emb_dims)

    def forward(self, x, turns):
        e1 = self.token_embedding(x)
        e2 = self.position_embedding(e1)
        e3 = self.turn_embedding(turns)
        e = e2 + e3
        return e


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=0.1):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, decoder=False):
        super().__init__() 
        # We set d_ff as a default to 2048
        if not decoder:
            self.linear_1 = nn.Linear(d_model, d_ff)
        else:
            self.linear_1 = nn.Linear(2*d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.gamma = nn.Parameter(torch.ones(self.size))
        self.beta = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.gamma * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.beta
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x_ = self.norm1(x)
        x = x + self.dropout(self.mha(x_, x_, x_, mask))
        x_ = self.norm2(x)
        output = x + self.dropout(self.ff(x_))
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(d_model, heads, dropout)
        self.mha_h = MultiHeadAttention(d_model, heads, dropout)
        self.mha_u = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout, decoder=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k, v, src_mask, tgt_mask, segment_type):
        x_ = self.norm1(x)
        m = x_ + self.dropout(self.mha1(x_, x_, x_, tgt_mask))
        q = self.norm2(x)
        mask_h = src_mask & (segment_type == 0).unsqueeze(1)
        c_h = m + self.dropout(self.mha_h(q, k, v, mask_h))
        c_h = self.norm3(c_h)
        mask_u = src_mask & (segment_type == 1).unsqueeze(1)
        c_u = m + self.dropout(self.mha_u(q, k, v, mask_u))
        c_u = self.norm4(c_u)
        c = torch.cat([c_h, c_u], dim=2)
        d = self.ff(c)
        return d, c_h, c_u


class Transformer(nn.Module):
    def __init__(self, d_model, heads, layers=6, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(layers)])
    
    def forward(self, src, tgt, src_mask, tgt_mask, segment_type):
        for encoder in self.encoders:
            src = encoder(src, src_mask)
        for decoder in self.decoders:
            tgt, c_h, c_u = decoder(tgt, src, src, src_mask, tgt_mask, segment_type)
        return src, tgt, c_h, c_u


class ReWritterModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, heads, layers, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dims, max_seq_len)
        self.transformer = Transformer(emb_dims, heads, layers, dropout)
        self.w_d = nn.Parameter(torch.randn((emb_dims, 1)))
        self.w_h = nn.Parameter(torch.randn((emb_dims, 1)))
        self.w_u = nn.Parameter(torch.randn((emb_dims, 1)))

    def forward(self, src_seqs, tgt_seqs, src_turns, tgt_turns, segment_type, transform_matrix):
        """[summary]

        Args:
            src_seqs (torch.tensor): (B x S)
            tgt_seqs (torch.tensor): (B x T)
            src_turns (torch.tensor): (B x S)
            segment_type (torch.tensor): (B x S) 
            transform_matrix (torch.tensor): (B x S x S)
            src_mask (torch.tensor): (B x S)
            tgt_mask (torch.tensor): (B x T)
        """
        src = self.embedding(src_seqs, src_turns)
        tgt = self.embedding(tgt_seqs, tgt_turns)
        src_mask, tgt_mask = self._compute_mask(src_seqs, tgt_seqs)
        src, tgt, c_h, c_u = self.transformer(src, tgt, src_mask, tgt_mask, segment_type)
        ratio = torch.sigmoid(torch.matmul(tgt, self.w_d) + torch.matmul(c_h, self.w_h) + torch.matmul(c_u, self.w_u)) # B X T x 1
        # ratio = torch.sigmoid(torch.matmul(tgt, self.w_d))
        scores = self._compute_scores(src, tgt, src_mask, segment_type)
        segment_type = segment_type.unsqueeze(1)  # B x 1 x S
        flag_h = (segment_type == 0) * 1.0
        flag_u = (segment_type == 1) * 1.0
        ratio = torch.bmm(ratio, flag_h) + torch.bmm((1 - ratio), flag_u)
        logits = ratio * scores
        logits = torch.bmm(logits, transform_matrix)
        return logits

    def _compute_scores(self, src, tgt, src_mask, segment_type):
        scores = torch.bmm(tgt, src.permute(0, 2, 1)) # B x T x S
        # segment_type = segment_type.unsqueeze(1)
        # scores1 = scores.masked_fill(~src_mask | (segment_type == 0), -1e9)
        # scores2 = scores.masked_fill(~src_mask | (segment_type == 1), -1e9)
        # scores1 = torch.softmax(scores1, dim=-1)
        # scores2 = torch.softmax(scores2, dim=-1)
        # scores = scores1 + scores2
        return scores

    def _compute_mask(self, src_seqs, tgt_seqs):
        src_mask = (src_seqs != 0).unsqueeze(1)
        tgt_mask = (tgt_seqs != 0).unsqueeze(1)
        size = tgt_seqs.size(1) # get seq_len for matrix
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(src_seqs.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def encode(self, src_seqs, src_turns):
        src_mask = (src_seqs != 0).unsqueeze(1)
        src = self.embedding(src_seqs, src_turns)
        for encoder in self.transformer.encoders:
            src = encoder(src, src_mask)
        return src
    
    def decode(self, src, src_mask, tgt_seqs, tgt_turns, segment_type, transform_matrix):
        tgt = self.embedding(tgt_seqs, tgt_turns)
        tgt_mask = (tgt_seqs != 0).unsqueeze(1)
        size = tgt_seqs.size(1) # get seq_len for matrix
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(src.device)
        tgt_mask = tgt_mask & nopeak_mask
        for decoder in self.transformer.decoders:
            tgt, c_h, c_u = decoder(tgt, src, src, src_mask, tgt_mask, segment_type)
        ratio = torch.sigmoid(torch.matmul(tgt, self.w_d) + torch.matmul(c_h, self.w_h) + torch.matmul(c_u, self.w_u)) # B X T x 1
        scores = self._compute_scores(src, tgt, src_mask,segment_type)
        segment_type = segment_type.unsqueeze(1)  # B x 1 x S
        flag_h = (segment_type == 0) * 1.0
        flag_u = (segment_type == 1) * 1.0
        ratio = torch.bmm(ratio, flag_h) + torch.bmm((1 - ratio), flag_u)
        scores = ratio * scores
        logits = torch.bmm(scores, transform_matrix.permute(0, 2, 1))
        # mask = abs(logits) < 1e-8
        # logits = logits.masked_fill(mask, -1e5)
        logp = torch.log_softmax(logits, dim=-1)
        return logp


if __name__ == "__main__":
    model = ReWritterModel(5, 10, 2, 2, 5)

    src = torch.tensor([[1, 3, 2, 0], [2, 1, 4, 0]], dtype=torch.long)
    tgt = torch.tensor([[1, 3, 2, 0, 0], [1, 2, 4, 0, 0]], dtype=torch.long)
    src_turns = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 2]], dtype=torch.long)
    tgt_turns = torch.tensor([[3, 3, 3, 3, 3], [3, 3, 3, 3, 3]], dtype=torch.long)
    segment_type = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.long)
    transform_matrix = torch.tensor([np.eye(4), np.eye(4)], dtype=torch.float)

    output = model(src, tgt, src_turns, tgt_turns, segment_type, transform_matrix)
    print(output)
    print(output.sum(-1))
    # print(torch.topk(output, 2))
