import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LstmRewriterModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, hidden_dims, class_weights=None, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        init_range = 0.5 / emb_dims
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.hidden_dims = hidden_dims
        self.bilstm = nn.LSTM(emb_dims, hidden_dims // 2, bidirectional=True, batch_first=True)
        self.W = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
        self.W_emb = nn.Parameter(torch.randn(emb_dims, emb_dims))
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()) # number of attention types, number of class
        self.out = nn.Linear(32, 3)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, contexts, utterances, labels=None):
        """[summary]

        Args:
            contexts (torch.tensor]): (b x ctx_len)
            utterances ([type]): b x utr_len
            labels ([type], optional): b x (ctx_len * utr_len)
        """
        ctx_emb = self.embedding(contexts)  
        utr_emb = self.embedding(utterances)
        ctx, utr = self._get_lstm_features(ctx_emb, utr_emb)
        attn_features = self._get_attn_features(ctx_emb, utr_emb, ctx, utr, contexts.shape[0])
        hidden = self.hidden(attn_features)
        logits = self.out(hidden)
        if labels is not None:
            logits = logits.view(-1, 3)
            labels = labels.view(-1)
            return self.loss(logits, labels)
            
        else:
            return logits

    def _init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dims // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dims // 2, device=device))
    
    def _get_lstm_features(self, ctx_emb, utr_emb):
        batch_size = ctx_emb.shape[0]
        ctx_len = ctx_emb.shape[1]
        emb = torch.cat([ctx_emb, utr_emb], dim=1)
        emb = self.dropout(emb)
        hidden = self._init_hidden(batch_size, ctx_emb.device)
        lstm_out, hidden = self.bilstm(emb, hidden)
        ctx = lstm_out[:, :ctx_len, :]
        utr = lstm_out[:, ctx_len:, :]
        return ctx, utr
    
    def _get_attn_features(self, ctx_emb, utr_emb, ctx, utr, batch_size):
        attn_features = []
        emb_dot = torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
        emb_bilinear = torch.matmul(ctx_emb, self.W_emb).bmm(utr_emb.permute(0, 2, 1)).unsqueeze(1)
        ctx_emb = F.normalize(ctx_emb)
        utr_emb = F.normalize(utr_emb)
        emb_cosine =  torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
        attn_features.append(emb_dot)
        attn_features.append(emb_bilinear)
        attn_features.append(emb_cosine)

        dot = torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        bilinear = torch.matmul(ctx, self.W).bmm(utr.permute(0, 2, 1)).unsqueeze(1)
        ctx = F.normalize(ctx)
        utr = F.normalize(utr)
        cosine =  torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        attn_features.append(dot)
        attn_features.append(bilinear)
        attn_features.append(cosine)

        num_attns = len(attn_features)
        attn_features = torch.cat(attn_features, dim=1)
        attn_features = attn_features.reshape(batch_size, num_attns, -1)
        attn_features = attn_features.permute(0, 2, 1)

        return attn_features
