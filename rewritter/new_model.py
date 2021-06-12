import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import AttentionUNet
from allennlp.modules.input_variational_dropout import InputVariationalDropout


class LstmRewriterModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, hidden_dims, class_weights=None, drop_in=0.2, drop_out=0.3, segment_type="fc"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        init_range = 0.5 / emb_dims
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.hidden_dims = hidden_dims
        self.bilstm = nn.LSTM(emb_dims, hidden_dims // 2, bidirectional=True, batch_first=True)
        # self.W = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
        self.W = nn.Linear(hidden_dims, hidden_dims)
        # self.W_emb = nn.Parameter(torch.randn(emb_dims, emb_dims))
        self.W_emb = nn.Linear(emb_dims, emb_dims)
        self.dropout_in = InputVariationalDropout(drop_in)
        self.dropout_out = InputVariationalDropout(drop_out)
        if segment_type == "fc":
            self.hidden = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()) # number of attention types, number of class
            self.out = nn.Linear(32, 3)
        else:
            self.unet = AttentionUNet(6, 3, 256)
        self.segment_type = segment_type
        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, contexts, utterances, labels=None):
        """

        Args:
            contexts (torch.tensor]): (b x ctx_len)
            utterances ([type]): b x utr_len
            labels ([type], optional): b x (ctx_len * utr_len)
        """
        ctx_mask = (contexts != 0).float()
        utr_mask = (utterances != 0).float()
        ctx_emb = self.embedding(contexts)  
        utr_emb = self.embedding(utterances)
        ctx, utr = self._get_lstm_features(ctx_emb, utr_emb, ctx_mask, utr_mask)
        attn_features = self._get_attn_features(ctx_emb, utr_emb, ctx, utr)
        if self.segment_type == "fc":
            hidden = self.hidden(attn_features)
            logits = self.out(hidden)
        else:
            batch_size = contexts.shape[0]
            segment_out = self.unet(attn_features)
            logits = segment_out.reshape(batch_size, -1, 3)
        if labels is not None:
            logits = logits.view(-1, 3)
            labels = labels.view(-1)
            return self.loss(logits, labels)
        else:
            return logits

    def _init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_dims // 2, device=device),
                torch.zeros(2, batch_size, self.hidden_dims // 2, device=device))
    
    def _get_lstm_features(self, ctx_emb, utr_emb, ctx_mask, utr_mask):
        batch_size = ctx_emb.shape[0]
        hidden = self._init_hidden(batch_size, ctx_emb.device)
        ctx, hidden = self.bilstm(self.dropout_in(ctx_emb), hidden)
        ctx = ctx * ctx_mask.unsqueeze(-1)
        ctx = self.dropout_out(ctx)
        hidden = self._init_hidden(batch_size, utr_emb.device)
        utr, hidden = self.bilstm(self.dropout_in(utr_emb), hidden)
        utr = utr * utr_mask.unsqueeze(-1)
        utr = self.dropout_out(utr)
        return ctx, utr

    def _get_attn_features(self, ctx_emb, utr_emb, ctx, utr):
        attn_features = []
        emb_dot = torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
        # emb_bilinear = torch.matmul(ctx_emb, self.W_emb).bmm(utr_emb.permute(0, 2, 1)).unsqueeze(1)
        emb_bilinear = torch.matmul(self.W_emb(ctx_emb), utr_emb.permute(0, 2, 1)).unsqueeze(1)
        ctx_emb = F.normalize(ctx_emb)
        utr_emb = F.normalize(utr_emb)
        emb_cosine =  torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
        attn_features.append(emb_dot)
        attn_features.append(emb_bilinear)
        attn_features.append(emb_cosine)

        dot = torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        # bilinear = torch.matmul(ctx, self.W).bmm(utr.permute(0, 2, 1)).unsqueeze(1)
        bilinear = torch.matmul(self.W(ctx), utr.permute(0, 2, 1)).unsqueeze(1)
        ctx = F.normalize(ctx)
        utr = F.normalize(utr)
        cosine =  torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        attn_features.append(dot)
        attn_features.append(bilinear)
        attn_features.append(cosine)
        num_attns = len(attn_features)
        attn_features = torch.cat(attn_features, dim=1)
        if self.segment_type == "fc":
            batch_size = ctx_emb.shape[0]
            attn_features = attn_features.reshape(batch_size, num_attns, -1)
            attn_features = attn_features.permute(0, 2, 1)

        return attn_features
