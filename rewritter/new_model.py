import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import AttentionUNet
from allennlp.modules.input_variational_dropout import InputVariationalDropout
import math
import torch
from . import utils
from transformers import AutoModel, AutoTokenizer


class LinearMatrixAttention(nn.Module):

    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                ) -> None:
        super().__init__()
        self._combination = combination
        combined_dim = utils.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = nn.Parameter(torch.Tensor(combined_dim))
        self._bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self,
                matrix_1: torch.Tensor,
                matrix_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = utils.combine_tensors_and_multiply(self._combination,
                                                             [matrix_1.unsqueeze(2), matrix_2.unsqueeze(1)],
                                                             self._weight_vector)
        return combined_tensors + self._bias


class RewriterModel(nn.Module):
    def __init__(self, emb_dims, hidden_dims, class_weights, vocab_size=None, dropout=0.2, 
            use_emb_attn=True, segment_type="fc") -> None:
        super().__init__()
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        self.W = nn.Parameter(torch.Tensor(hidden_dims, hidden_dims))
        torch.nn.init.xavier_uniform_(self.W)
        self.linear = LinearMatrixAttention(hidden_dims, hidden_dims, "x,y")
        if use_emb_attn:
            self.W_emb = nn.Parameter(torch.Tensor(emb_dims, emb_dims))
            torch.nn.init.xavier_uniform_(self.W_emb)
            self.emb_linear = LinearMatrixAttention(emb_dims, emb_dims, 'x,y')
            num_attns = 8
        else:
            num_attns = 4
        self.use_emb_attn = use_emb_attn
        self.dropout = nn.Dropout(dropout)
        if segment_type == "fc":
            self.hidden = nn.Sequential(nn.Linear(num_attns, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            self.out = nn.Linear(16, 3)
        elif segment_type == "unet":
            self.unet = AttentionUNet(num_attns, 3, 256)
        self.segment_type = segment_type
        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    def forward(self, contexts, utterances, labels=None):
        """[summary]

        Args:
            contexts (torch.tensor]): (b x ctx_len)
            utterances ([type]): b x utr_len
            labels ([type], optional): b x (ctx_len * utr_len)

        Returns:
            [type]: [description]
        """
        ctx_emb, utr_emb, ctx, utr = self._get_embedding_and_encoding(contexts, utterances)
        attn_features = self._get_attn_features(ctx, utr, ctx_emb, utr_emb)
        if self.segment_type == "fc":
            hidden = self.hidden(attn_features)
            logits = self.out(hidden)
        elif self.segment_type == "unet":
            if not hasattr(contexts, "shape"):
                batch_size = contexts["input_ids"].shape[0]
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
    
    def _get_embedding_and_encoding(self, contexts, utterances):
        pass

    def _get_embedding(self, contexts, utterances):
        ctx_emb = self.embedding(contexts)  
        utr_emb = self.embedding(utterances)
        emb = torch.cat([ctx_emb, utr_emb], dim=1)
        emb = self.dropout(emb)
        ctx_len = contexts.shape[1]
        ctx_emb = emb[:, :ctx_len, :]
        utr_emb = emb[:, ctx_len:, :]
        return ctx_emb, utr_emb

    def _get_attn_features(self, ctx, utr, ctx_emb=None, utr_emb=None):
        attn_features = []
        dot = torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        bilinear = torch.matmul(ctx, self.W).bmm(utr.permute(0, 2, 1)).unsqueeze(1)
        linear = self.linear(ctx, utr).unsqueeze(1)
        ctx = F.normalize(ctx, dim=-1)
        utr = F.normalize(utr, dim=-1)
        cosine =  torch.bmm(ctx, utr.permute(0, 2, 1)).unsqueeze(1)
        attn_features.append(dot)
        attn_features.append(bilinear)
        attn_features.append(cosine)
        attn_features.append(linear)

        if ctx_emb is not None and utr_emb is not None:
            emb_dot = torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
            emb_bilinear = torch.matmul(ctx_emb, self.W_emb).bmm(utr_emb.permute(0, 2, 1)).unsqueeze(1)
            emb_linear = self.emb_linear(ctx_emb, utr_emb).unsqueeze(1)
            ctx_emb = F.normalize(ctx_emb, dim=-1)
            utr_emb = F.normalize(utr_emb, dim=-1)
            emb_cosine =  torch.bmm(ctx_emb, utr_emb.permute(0, 2, 1)).unsqueeze(1)
            attn_features.append(emb_dot)
            attn_features.append(emb_bilinear)
            attn_features.append(emb_linear)
            attn_features.append(emb_cosine)

        num_attns = len(attn_features)
        attn_features = torch.cat(attn_features, dim=1)
        if self.segment_type == "fc":
            batch_size = ctx.shape[0]
            attn_features = attn_features.reshape(batch_size, num_attns, -1)
            attn_features = attn_features.permute(0, 2, 1)

        return attn_features


class LstmRewriterModel(RewriterModel):
    def __init__(self, vocab_size, emb_dims, hidden_dims, class_weights=None, dropout=0.2, segment_type="fc"):
        super().__init__(emb_dims, hidden_dims, class_weights, vocab_size, dropout, True, segment_type)
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        # init_range = 0.5 / emb_dims
        # self.embedding.weight.data.uniform_(-init_range, init_range)
        self.hidden_dims = hidden_dims
        self.bilstm = nn.LSTM(emb_dims, hidden_dims // 2, bidirectional=True, batch_first=True)

    def _get_embedding_and_encoding(self, contexts, utterances):
        ctx_mask = contexts != 0
        utr_mask = utterances != 0
        ctx_emb, utr_emb = self._get_embedding(contexts, utterances)
        ctx, utr = self._get_lstm_features(ctx_emb, utr_emb, ctx_mask, utr_mask)
        return ctx_emb, utr_emb, ctx, utr

    def _init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_dims // 2, device=device),
                torch.zeros(2, batch_size, self.hidden_dims // 2, device=device))

    def _get_lstm_features(self, ctx_emb, utr_emb, ctx_mask, utr_mask):
        batch_size = ctx_emb.shape[0]
        hidden = self._init_hidden(batch_size, ctx_emb.device)
        ctx, hidden = self.bilstm(ctx_emb, hidden)
        # ctx = ctx * ctx_mask.unsqueeze(-1).float()
        hidden = self._init_hidden(batch_size, utr_emb.device)
        utr, hidden = self.bilstm(utr_emb, hidden)
        # utr = utr * utr_mask.unsqueeze(-1).float()
        return ctx, utr


class TransformerRewriterModel(RewriterModel):
    def __init__(self, vocab_size, emb_dims, heads=10, layers=1, class_weights=None, 
            dropout=0.2, segment_type="fc"):
        super().__init__(emb_dims, emb_dims, class_weights, vocab_size, dropout, True, segment_type)
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        # init_range = 0.5 / emb_dims
        # self.embedding.weight.data.uniform_(-init_range, init_range)
        encoder_layer = nn.TransformerEncoderLayer(emb_dims, heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
    
    def _get_embedding_and_encoding(self, contexts, utterances):
        ctx_mask = contexts == 0
        utr_mask = utterances == 0
        ctx_emb, utr_emb = self._get_embedding(contexts, utterances)
        emb = torch.cat([ctx_emb, utr_emb], dim=1)
        mask = torch.cat([ctx_mask, utr_mask], dim=1)
        transformed = self.transformer(emb.transpose(0, 1), src_key_padding_mask=mask)
        transformed = transformed.transpose(0, 1)
        ctx_len = contexts.shape[1]
        ctx = transformed[:, :ctx_len, :]
        utr = transformed[:, ctx_len:, :]
        return ctx_emb, utr_emb, ctx, utr


class BertRewriterModel(RewriterModel):
    def __init__(self, model_name_or_path, class_weights=None, dropout=0.2, segment_type="fc"):
        super().__init__(768, 768, class_weights, None, dropout, False, segment_type)
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def _get_embedding_and_encoding(self, contexts, utterances):
        inputs = {key: torch.cat([contexts[key], utterances[key]], dim=1) for key in contexts}
        output = self.bert_model(**inputs)
        encoding = output.last_hidden_state
        ctx_len = contexts["input_ids"].shape[1]
        ctx = encoding[:, :ctx_len, :]
        utr = encoding[:, ctx_len:, :]
        return  None, None, ctx, utr
