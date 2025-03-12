"""An autoregressive Transformer.

This implementation allows for an arbirary ordering of input variables; the
appropriate masking is automatically calculated.
"""
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchviz import make_dot

import common
import masking
from model.made import seqList_2_EnsembleSeq
from utils import util
from utils.util import get_device

## Notes on masking schemes
#
# Supported values of MASK_SCHEME: 0 (only works for natural ordering), 1 (for
# arbitrary ordering).
#
# Assume we have (x0, x1, x2) and order=natural in all cases below.
#
# Scheme 0 (only works for natural ordering):
#         Input  [SOS, x0, x1]
#         Hidden [p(x0), p(x1|x0), p(x2|x0,x1)]
#      (We use 1 block so hidden == output).
#
#      mask(3) is used with the following result (row=destination index):
#         tensor([[1, 0, 0],
#                 [1, 1, 0],
#                 [1, 1, 1]], dtype=torch.uint8)
#
# Scheme 1:
#   Here, the first attention layer has a different mask than the subsequent
#   layers.  See the detailed docstring of order_respecting_mask().
#
## Notes on unk_embeddings & pos_embeddings
#
#  pos_embeddings index: *position* of input sequence
#    - aganostic to what column index it is (or even if it is SOS)
#
#  unk_embeddings index: *natural_idx* of the column it's supposed to mask
#    - thus, SOS does not have an unk_embeddings
#
#  How they interact: potentially dropout first, then potentially add pos emb.

MASK_SCHEME = 1
# MASK_SCHEME = 0


def mask(n):
    ns = n
    nd = n
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns)
    m = i >= j - ns + nd
    m.requires_grad = False
    return m


def order_respecting_mask(ncols, ordering, input_layer=True):
    """Construct appropriate mask for attention.

    Assuming o=(2,0,1):
     - set inputs = [ SOS=0,          x0,    x1,     x2 ]
     - so outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]

    No one connects to EOS.  SOS connects to everyone.

    Desired mask (row=destination):
        [[1, 0, 0, 1],
         [1, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 0]]

    Mask after the first attention + see self (diagonal)
    Basically == shift above to the left 1 column, then fill diagonal
     - inputs  = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
     - outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
        [[1, 0, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]]
    """
    mask = np.zeros((ncols + 1, ncols + 1))

    if input_layer:
        mask[:, 0] = 1  # First column is SOS -- everyone can see.
        mask[-1, :] = 0  # No one connects to EOS.
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src + 1, ncols):
                # Variable at pos_dst should see pos_src.
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx + 1] = 1
    else:
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src, ncols):
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx] = 1

    mask = torch.as_tensor(mask, dtype=torch.float32)
    mask.requires_grad = False
    return mask


class LayerNorm(nn.Module):
    """Norm to 0-mean 1-std , then do a learned diagonal affine transform."""

    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        s = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(s + self.eps)
        return self.scale * x + self.shift


class Conv1d(nn.Module):
    """Linear with bias add.  Weights ~ N(std), bias ~ 0."""

    def __init__(self, d_in, d_out, w_init_std=0.02):
        super(Conv1d, self).__init__()

        self.w = nn.Parameter(torch.zeros(d_in, d_out))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.w, std=w_init_std)
        nn.init.zeros_(self.b)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):
        *start, d_in = x.size()
        out = torch.matmul(x.view(-1, d_in), self.w) + self.b
        return out.view(start + [self.d_out])


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer.

    Args:
      d_model: last dim of input and output of this module.
      num_heads: number of parallel heads.

    Internally, queries, keys, and values are all produced from the input
    (hence "self"), and all of them are (d_model/num_heads)-dimensional.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_state = d_model // num_heads

        self.qkv_linear = Conv1d(d_model, self.d_state * 3 * num_heads)
        self.linear = Conv1d(num_heads * self.d_state, d_model)

        self.attn_mask = None  # Will be set by caller.

    def _split_heads(self, x):
        # Each input has shape [bs, num cols, d_state * num_heads].
        *start, m = x.size()
        x = x.view(start + [self.num_heads, m // self.num_heads])
        return x.permute(0, 2, 1, 3)

    def _do_attention(self, query, key, value, mask):
        """Accepts Q,K,V each shaped [bs, num heads, num cols, d_state].

        Returns transformed [bs, num_heads, num cols, d_state].
        """
        d_k = query.size()[-1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)
        mask = mask.to(scores.dtype)
        scores = scores * mask - (1 - mask) * 1e10
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, value)
        return out

    def forward(self, x, query_input=None):
        """x: [bs, num cols, d_model].  Output has the same shape."""
        assert x.dim() == 3, x.size()
        bs, ncols, _ = x.size()

        # [bs, num cols, d_state * 3 * num_heads]
        qkv = self.qkv_linear(x)
        # [bs, num heads, num cols, d_state] each
        qs, ks, vs = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        if query_input is not None:
            # assert Exception('dont support query_input when supporting factorized table')
            # TODO: obviously can avoid redundant calc.
            qkv = self.qkv_linear(query_input)
            qs, _, _ = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        # [bs, num heads, num cols, d_state]
        x = self._do_attention(qs, ks, vs, mask=self.attn_mask.to(x.device))

        # [bs, num cols, num heads, d_state]
        x = x.transpose(1, 2)
        # Concat all heads' outputs: [bs, num cols, num heads * d_state]
        x = x.contiguous().view(bs, ncols, -1)
        # Then do a transform: [bs, num cols, d_model].
        x = self.linear(x)
        return x


class Block(nn.Module):
    """A Transformer block.

    Args:
      d_model: last dim of input and output of this module.
      d_ff: the hidden dim inside the FF net.
      num_heads: number of parallel heads.
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 num_heads,
                 activation='relu',
                 do_residual=False,
                 dropout=0):
        super(Block, self).__init__()

        self.mlp = nn.Sequential(
            Conv1d(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            Conv1d(d_ff, d_model),
        )
        self.norm1 = LayerNorm(features=d_model)
        self.norm2 = LayerNorm(features=d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.do_residual = do_residual

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def set_attn_mask(self, mask):
        self.attn.attn_mask = mask

    def forward(self, x, query_input=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, query_input=query_input)
        x = self.dropout1(x)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        if self.do_residual:
            x += residual

        return x


class Transformer(nn.Module):
    """An autoregressive Transformer."""

    def __init__(
            self,
            num_blocks,
            d_model,
            d_ff,
            num_heads,
            nin,
            input_bins,
            use_positional_embs=True,
            activation='gelu',
            dropout=False,
            fixed_ordering=None,
            draw_dropout_per_col=False,
            per_row_dropout=False,
            seed=None,
            ln_output=True,
            transformer_dropout=0,
            factor_table=None,
            use_ensemble=True,
            multi_pred_embedding='mlp',
            # Join support.
            join_args={},
    ):
        """An autoregressive Transformer.

        Namings of the arguments follow from the original paper.

        Args:
          num_blocks: int, number of transformer blocks.
          d_model: int, the hidden dims.
          d_ff: int, each feedforward layer's hidden dims.
          num_heads: int, number of attention heads in each self-attention.
          nin: int, number of input variables.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          use_positional_embs: bool, whether positional encodings are used
            (i.e., whether an input is treated as a sequence or as a set).
          activation: str, the activation function.
          dropout: if True, turn on column masking during training time, which
            enables the wildcard skipping (aka variable skipping) optimization
            during inference.  Recommended to be set for any non-trivial
            datasets.
          fixed_ordering: variable ordering to use.  Ex: [2, 0, 1] means
            variable 2 is placed in the first position in the autoregressive
            factorization.  If None, either natural ordering (when seed is
            None) or a randomly sampled ordering (otherwise) is used.
          draw_dropout_per_col: bool.
          seed: if specified, used for sampling a random ordering.
          ln_output: if set, layer normalization is applied to the final output
            of the Transformer stack.
          transformer_dropout: dropout probability inside the Transformer stack.
          join_args: dict containing join-support arguments.
        """
        super().__init__()

        print('MASK_SCHEME', MASK_SCHEME)

        # Common attributes below.
        self.nin = nin
        self.input_bins = input_bins
        encoded_bins = [d_model] * nin
        self.logit_indices = np.cumsum(encoded_bins)
        self.nout = self.logit_indices[-1]
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.embed_size = d_model
        self.emb_dim = d_model
        self.use_positional_embs = use_positional_embs
        assert activation in ['relu', 'gelu']
        self.activation = activation
        self.factor_table = factor_table
        self.original_nin = self.nin if factor_table is None else len(factor_table.base_table.columns)
        self.original_input_bins = input_bins if factor_table is None else [c.distribution_size for c in factor_table.base_table.columns]
        self.fixed_ordering = fixed_ordering
        self.multi_pred_embedding = multi_pred_embedding
        if fixed_ordering is None:
            natural = np.arange(nin)
            if seed is None:
                self.fixed_ordering = natural
            else:
                self.fixed_ordering = np.random.RandomState(seed).permutation(
                    natural)
        print('ordering', self.fixed_ordering)
        self.draw_dropout_per_col = draw_dropout_per_col
        self.per_row_dropout = per_row_dropout
        self.ln_output = ln_output
        self.transformer_dropout = transformer_dropout

        # Masking.
        p = masking.Masking.Params()
        p['draw_dropout_per_col'] = self.draw_dropout_per_col
        p['per_row_dropout'] = self.per_row_dropout
        p.update(join_args)
        self.masking = masking.Masking(p)

        # Build.
        self.blocks = nn.Sequential(*[
            Block(d_model,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=(MASK_SCHEME == 0 or i > 0),
                  dropout=transformer_dropout) for i in range(num_blocks)
        ])
        # Set masks.
        self.update_masks(self.fixed_ordering)

        if ln_output:
            self.norm = LayerNorm(d_model)

        self.embeddings = nn.ModuleList()
        for i in range(self.nin):
            self.embeddings.append(nn.Embedding(self.input_bins[i] + 5, d_model))
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.02)
        self.embeddings_out = nn.ModuleList()
        for i in range(self.nin):
            self.embeddings_out.append(nn.Embedding(self.input_bins[i], d_model))
        for e in self.embeddings_out:
            nn.init.normal_(e.weight, std=0.02)

        if use_positional_embs:
            if MASK_SCHEME == 1:
                self.pos_embeddings = nn.Embedding(self.nin + 1, d_model)
            else:
                self.pos_embeddings = nn.Embedding(self.nin, d_model)
            nn.init.normal_(self.pos_embeddings.weight, std=0.01)

        self.dropout = dropout
        self.unk_embeddings = nn.ParameterList()
        for i, dist_size in enumerate(self.input_bins):
            self.unk_embeddings.append(nn.Parameter(torch.zeros(d_model)))

        # Interface required by ProgressiveSampling.
        self.input_bins_encoded_cumsum = np.cumsum(encoded_bins)
        self.use_ensemble = use_ensemble
        self.multi_pred_embed_nn = None
        self.is_ensembled = False
        if self.multi_pred_embedding == 'mlp':
            net_num = self.nin if MASK_SCHEME == 0 else self.nin + 1
            # if use_ensemble:
            #     self.multi_pred_embed_nn = seqList_2_EnsembleSeq([nn.Sequential(
            #         nn.Linear(d_model * 2, 64),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(64, d_model)
            #     ) for i in range(net_num)])
            #     self.is_ensembled = True
            # else:
            self.multi_pred_embed_nn = [nn.Sequential(
                nn.Linear(d_model * 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, d_model)
            ) for i in range(net_num)]
            self.multi_pred_embed_nn = nn.ModuleList(self.multi_pred_embed_nn)
        else:
            assert Exception('not supported yet')
        # self.predicate_mapping_net = None
        # if factor_table is not None and isinstance(factor_table, common.FactorizedTable):
        #     self.predicate_mapping_net = []
        #     for i in range(self.original_nin):
        #         # original_col_name = self.factor_table.base_table.columns[i].name
        #         # fact_cols = self.factor_table.map_from_col_to_fact_col[original_col_name]
        #         fact_cols_net = []
        #         for j in range(factor_table.fact_col_number[i]):
        #             fact_cols_net.append(
        #                 nn.Sequential(
        #                     nn.Linear(d_model, 64),
        #                     nn.ReLU(inplace=True),
        #                     nn.Linear(64, d_model)
        #                 )
        #             )
        #         self.predicate_mapping_net.append(nn.ModuleList(fact_cols_net))
        #     self.predicate_mapping_net = nn.ModuleList(self.predicate_mapping_net)
        self.orderings = [self.fixed_ordering]

    def ConvertToEnsemble(self):
        pass

    def ConvertToUnensemble(self):
        pass

    def update_masks(self, ordering=None):
        ordering = self.fixed_ordering if ordering is None else ordering
        self.fixed_ordering = ordering
        orig_mask = None
        if MASK_SCHEME == 0:
            assert (self.fixed_ordering == np.arange(len(self.fixed_ordering))).all(), f'only support natural ordering, given order is {self.fixed_ordering}'
            orig_mask = mask(self.nin)
        elif MASK_SCHEME == 1:
            init_attn_mask = order_respecting_mask(self.nin, self.fixed_ordering)
            attn_mask = order_respecting_mask(self.nin,
                                              self.fixed_ordering,
                                              input_layer=False)
        else:
            assert False, MASK_SCHEME

        if orig_mask is not None:
            for b in self.blocks:
                b.set_attn_mask(orig_mask)
        else:
            self.blocks[0].set_attn_mask(init_attn_mask)
            for b in self.blocks[1:]:
                b.set_attn_mask(attn_mask)

    def name(self):
        n = 'transformer'
        n += '-blocks' + str(self.num_blocks)
        n += '-model' + str(self.d_model)
        n += '-ff' + str(self.d_ff)
        n += '-heads' + str(self.num_heads)
        if self.use_positional_embs:
            n += '-posEmb'
        n += '-' + self.activation
        if self.dropout:
            n += '-dropout'
        if MASK_SCHEME == 1:
            n += '-scheme1'
        return n

    def UseDMoL(self, natural_idx):
        """Returns True if we want to use DMoL for this column."""
        return False

    def EncodeInput(self, x, preds, natural_col=None, out=None, train=False):
        # x->[bs, 2, ncols]
        """Right shift by one token.

        Suppose we want to model x=(x0,x1,x2).
        Set model inputs = [ SOS=0, x0, x1 ]
            (SOS = start of sequence)
        outputs =          [ p(x0); p(x1|x0); p(x2|x0,x1) ].
            (because output i depends on inputs <= i).

        If self.fixed_ordering is supplied and non-natural,
        we set inputs = [ SOS=0, x_o(0), x_o(1) ]
        so    outputs = [ p(x_o(0)), p(x_o(1) | x_o(0)), p(x_o(2) | x_o(0..1)) ]

        This (1) requires when calculating the loss, seq [x_o(0), ..., x_o(2)]
        is passed, (2) assumes we don't change the diagonal attention mask.

        Alternatively (assuming o=(2,0,1)):
          - change diagonal mask to respect ordering o
          - set inputs = [ SOS=0, x_o(0)=x2, x_o(1)=x0 ]
          - so outputs = [ p(x0|x2), p(x1|x0,x2), p(x2) ]
          - doesn't require feeding targets under order o
        """
        # re-arange predicates order to give a fixed and specific order of predicate operator
        data, preds = util.sortPredicates(x, preds)
        # if natural_col is not None:
        #     # if x is None:
        #     #     # preds = torch.zeros((out.shape[0], 5), device=out.device)
        #     #     preds[..., 0] = 1
        #     return self.EncodeInputInference(x, preds, natural_col, out)

        if x.dtype != torch.long:
            x = x.long()
        bs = x.size()[0]
        none_pred_masks = [torch.zeros(bs, 2, device=x.device)]
        if MASK_SCHEME == 0:
            # SOS = start of sequence symbol, just zeros.
            y_embed = [torch.zeros(bs, 2, self.embed_size, device=x.device)]
            for nat_idx in range(self.nin - 1):
                data_i = x[..., nat_idx]
                none_pred_mask = (data_i == -1).to(torch.float32, non_blocking=True, copy=False)
                none_pred_masks.append(none_pred_mask)
                data_i = torch.where(data_i == -1, 0, data_i)
                y_embed.append(torch.matmul(
                    torch.cat([torch.zeros(x.shape[0], 2, self.input_bins[nat_idx], device=x.device).scatter(-1, data_i.view(
                        -1, 2, 1), 1), preds[..., nat_idx * 5:(nat_idx + 1) * 5]], dim=-1), self.embeddings[nat_idx].weight))
        elif MASK_SCHEME == 1:
            y_embed = [torch.zeros(bs, 2, self.embed_size, device=x.device)]
            for nat_idx in range(self.nin):
                data_i = x[..., nat_idx]
                none_pred_mask = (data_i == -1).to(torch.float32, non_blocking=True, copy=False)
                none_pred_masks.append(none_pred_mask)
                data_i = torch.where(data_i == -1, 0, data_i)
                y_embed.append(torch.matmul(
                    torch.cat([torch.zeros(x.shape[0], 2, self.input_bins[nat_idx], device=x.device).scatter(-1, data_i.view(
                        -1, 2, 1), 1), preds[..., nat_idx * 5:(nat_idx + 1) * 5]], dim=-1), self.embeddings[nat_idx].weight))
        else:
            assert False, MASK_SCHEME
        none_pred_masks = torch.stack(none_pred_masks, dim=-1).unsqueeze(-1)

        # [batch size, 2, num cols (+ 1), d_model].  +1 or not depends on scheme.
        inp = torch.stack(y_embed, 2)
        dropped_repr = torch.tile(torch.stack(tuple(self.unk_embeddings)).unsqueeze(0).unsqueeze(1), dims=(1, 2, 1, 1))
        # Shaped [1, 2, num cols, d_model].
        if MASK_SCHEME == 0:
            # Namely, [0, unk(0), unk(1)] for ncols=3.  This means:
            #   (1) SOS is never dropped.
            #   (2) indexing into unk_embeddings is based on natural_idx.
            dropped_repr = torch.cat((torch.zeros_like(
                dropped_repr[..., 0:1, :]), dropped_repr[..., :-1, :]),
                dim=2)
        else:
            dropped_repr = torch.cat(
                (torch.zeros_like(dropped_repr[..., 0:1, :]), dropped_repr),
                dim=2)
        if train and self.dropout:
            batch_mask = self.masking.input_mask(x, is_training=self.training)
            if MASK_SCHEME == 1:
                batch_mask = torch.concat([torch.ones((batch_mask.shape[0], 2, 1, batch_mask.shape[-1]), device=x.device), batch_mask], dim=2)
            inp = batch_mask * inp + (1. - batch_mask) * dropped_repr
        inp = dropped_repr * none_pred_masks + (1. - none_pred_masks) * inp

        return inp

    def EncodeInputInference(self, x, preds, natural_col, out):
        """Special inference path.

        Args:
          x: [batch size, 1].  Just the data for column 'natural_col'.
          natural_col (int): [0, num cols).
          out: shaped [batch size, d_model].  To hold the encoded data.
        """
        if natural_col < 0:
            # Potentially handling SOS.
            if self.use_positional_embs:
                # Let's also add E_pos=0 to SOS (if enabled).
                out.copy_(self.pos_embeddings(torch.as_tensor(
                        0,
                        device=get_device())).unsqueeze(0).unsqueeze(1).expand(out.size()[0], 2, -1))
            return
        dropped_repr = self.unk_embeddings[natural_col].unsqueeze(0).unsqueeze(1).expand(
            out.shape[0], 2, -1)
        if x is None:
            # [bs, 2, d_model]
            embs = dropped_repr
        else:
            # [bs, 2, d_model]
            none_pred_mask = (x == -1).to(torch.float32, non_blocking=True, copy=False)
            x = torch.where(x == -1, 0, x)
            embs = torch.matmul(
                torch.cat(
                    [torch.zeros(x.shape[0], 2, self.input_bins[natural_col], device=x.device).scatter(-1, x.long(), 1),
                     preds], dim=-1), self.embeddings[natural_col].weight)
            embs = dropped_repr * none_pred_mask + (1. - none_pred_mask) * embs

        if self.use_positional_embs:
            # NOTE: this is tricky.  Under MASK_SCHEME=0 or 1, E_pos=0 is added
            # to SOS, E_pos=1 is added to x0, etc.  So we need to take this into
            # account.
            pos = self.pos_embeddings(
                torch.as_tensor(natural_col + 1,
                                device=out.device)).unsqueeze(0)
            embs = embs + pos.unsqueeze(1)

        out.copy_(embs)

    def merge_multi_preds(self, x):
        assert self.multi_pred_embedding == 'mlp'
        # 与MADE不同的方法，这里是直接拼接后输入MLP，而不是分别过MLP再求平均
        '''
        if MASK_SCHEME == 0:
            # x: [bs, 2, ncols, d_model]
            if self.multi_pred_embedding == 'mlp' and self.is_ensembled:
                # [bs, 2, ncols, d_model] -> [bs, ncols, 2, d_model] -> [bs, ncols*2*d_model] -> [bs, ncols*d_model]->[bs, ncols, d_model]
                return torch.stack(torch.split(self.multi_pred_embed_nn(x.transpose(1, 2).flatten(1, 3)), self.d_model, -1), 1)
            out = []
        else:
            # x: [bs, 2, 1+ncols, d_model] for training, [bs, 2, d_model] for inference
            begin = x[..., 0, :]
            x = x[..., 1:, :]
            if self.multi_pred_embedding == 'mlp' and self.is_ensembled:
                # [bs, 2, ncols, d_model] -> [bs, ncols, 2, d_model] -> [bs, ncols*2*d_model] -> [bs, ncols*d_model]->[bs, ncols, d_model]
                tmp = torch.stack(torch.split(self.multi_pred_embed_nn(x.transpose(1, 2).flatten(1, 3)), self.d_model, -1), 1)
                return torch.cat([begin, tmp], dim=1)
            out = [begin]
        '''
        if self.is_ensembled:
            # [bs, 2, ncols, d_model] -> [bs, ncols, 2, d_model] -> [bs, ncols*2*d_model] -> [bs, ncols*d_model]->[bs, ncols, d_model]
            return torch.stack(torch.split(self.multi_pred_embed_nn(x.transpose(1, 2).contiguous().flatten(1, 3)), self.d_model, -1), 1)
        out = []
        for i in range(0, self.nin if MASK_SCHEME == 0 else self.nin + 1):
            # [bs, 2, d_model] -> [bs, 2*d_model] -> [bs, d_model]
            out.append(self.multi_pred_embed_nn[i](x[:, :, i, :].flatten(1, 2)))
        return torch.stack(out, dim=1)

    def mapping_factorized_preds(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape=(bs, ncols, d_model)
        """
        if self.predicate_mapping_net is not None:
            res = [] if MASK_SCHEME == 0 else [x[:, 0, :]]
            for i in range(self.original_nin):
                for j in range(self.factor_table.fact_col_number[i]):
                    res.append(self.predicate_mapping_net[i][j](x[:, i if MASK_SCHEME == 0 else i + 1, :]))
            return torch.stack(res, dim=1)
        else:
            return x

    def forward(self, x, train=False):
        """Outputs logits for (x0, x1|x0, x2|x0,x1, ...)."""
        # [bs, 2, ncols] -> [bs, ncols, d_model].  Right-shifted.

        # [bs, 2, ncols] -> [bs, 2, ncols, d_model]
        x = self.EncodeInput(*x, train=train)
        # [bs, 2, ncols, d_model] -> [bs, ncols, d_model]
        x = self.merge_multi_preds(x)
        # x = self.mapping_factorized_preds(x)
        if self.use_positional_embs:
            # [1, inp_seq_len, d_model]
            # NOTE: indexes into pos embs == positions \in [0, inp_seq_len).
            pos_embs = self.pos_embeddings(
                torch.arange(x.shape[-2], device=x.device)).unsqueeze(0)
            x += pos_embs
        if MASK_SCHEME == 1:
            assert self.use_positional_embs, 'should enable positional embs'
            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x = self.blocks(x)

        if self.ln_output:
            x = self.norm(x)
        return x

    def forward_with_encoded_input(self, x):
        x = self.merge_multi_preds(x)
        x = self.mapping_factorized_preds(x)
        if self.use_positional_embs:
            # [1, inp_seq_len, d_model]
            # NOTE: indexes into pos embs == positions \in [0, inp_seq_len).
            pos_embs = self.pos_embeddings(
                torch.arange(x.shape[-2], device=x.device)).unsqueeze(0)
            x += pos_embs
        # [batch size, num cols * d_model] -> [bs, num cols, d_model]
        x = x.view(x.shape[0], -1, self.d_model)

        if MASK_SCHEME == 1:
            inp_seq_len = x.shape[1]

            assert self.use_positional_embs, 'should enable positional embs'
            # pos_embs = torch.tile(self.pos_embeddings(
            #     torch.arange(inp_seq_len, device=x.device)).unsqueeze(0).unsqueeze(1), dims=(1, 2, 1, 1))

            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x = self.blocks(x)

        if self.ln_output:
            x = self.norm(x)
        return x

    def nll(self, logits, data, table:Union[common.TableDataset, common.FactorizedTable], label_smoothing=0, use_weight=False):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, ncols+1, d_model].
          data: [batch size, ncols].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            if label_smoothing == 0:
                loss = F.cross_entropy(logits_i, data[:, i], reduction='none', weight=table.columns[i].dist_weights.to(logits.device) if use_weight and table.columns[i].dist_weights is not None else None)
            else:
                log_probs_i = logits_i.log_softmax(-1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_probs_i)
                    true_dist.fill_(label_smoothing / (self.input_bins[i] - 1))
                    true_dist.scatter_(1, data[:, i].unsqueeze(1),
                                       1.0 - label_smoothing)
                loss = (-true_dist * log_probs_i).sum(-1)
            nll += loss
        return nll

    def logits_for_col(self, idx, logits, out=None, reduce=False):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, ncols+1, d_model].

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        logits_for_var = logits[:, idx, :]
        embed = self.embeddings_out[idx]
        if reduce:
            logits_for_var = logits_for_var[0]
        return torch.matmul(torch.exp(logits_for_var), embed.weight.t(), out=out)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    num_cols = 3
    vocab = 1
    bs = 1
    num_cols = 11
    vocab = 5
    bs = 100
    orderings = [
        np.arange(num_cols),
        np.arange(num_cols)[::-1],
        np.random.permutation(np.arange(num_cols)),
    ]
    for ordering in orderings:
        print('Testing ordering', ordering)
        model = Transformer(num_blocks=1,
                            d_model=16,
                            d_ff=64,
                            num_heads=4,
                            nin=num_cols,
                            input_bins=[
                                           vocab,
                                       ] * num_cols,
                            use_positional_embs=True,
                            activation='gelu',
                            fixed_ordering=ordering,
                            dropout=True)
        print('attn_mask for blk 0', model.blocks[0].attn.attn_mask)

        for i in range(num_cols):
            nat_idx = ordering[i]
            print('\nchecking output column {} nat_idx {}...'.format(
                i, nat_idx))
            inp = torch.randint(vocab, (bs, 2, num_cols))
            # preds = torch.tile(torch.tensor([[1., 0., 0., 0., 0.] * num_cols], requires_grad=True).unsqueeze(1),
            #                    dims=(1, 2, 1))
            preds = torch.tile(torch.tensor([[1., 0., 0., 0., 0.] * num_cols], requires_grad=True).unsqueeze(1), dims=(inp.shape[0], 2, 1))
            # [bs, num cols, d_model], the logits
            for p in model.parameters():
                p.retain_grad = True
            out = model([inp, preds])
            out[:, nat_idx, :].contiguous().mean().backward(retain_graph=True)
            # dot = make_dot(out[:, nat_idx, :].contiguous().view(-1, )[0], params=dict(model.named_parameters()), show_attrs=True, show_saved=True, max_attr_chars=100000, display_var_set={"data",
            #                                                                                                                                                                                "device",
            #                                                                                                                                                                                "dtype",
            #                                                                                                                                                                                "grad",
            #                                                                                                                                                                                "grad_fn",
            #                                                                                                                                                                                "is_cpu",
            #                                                                                                                                                                                "is_cuda",
            #                                                                                                                                                                                "is_leaf",
            #                                                                                                                                                                                "requires_grad",
            #                                                                                                                                                                                "retains_grad"})
            # dot.format = 'svg'
            # dot.render()
            ok = True
            connected = True
            for n, p in model.named_parameters():
                if 'embed' in n:
                    if p.grad is None:
                        print(n, p.grad)
                        continue
                    dep = (p.grad.reshape(-1) != 0).numpy().any()
                    for j in range(i, len(ordering)):
                        nat_idx_j = ordering[j]
                        # i.e., p corresponds to nat_idx j
                        if n == 'embeddings.{}.weight'.format(nat_idx_j):
                            ok &= (not dep)
                    for j in range(0,i):
                        nat_idx_j = ordering[j]
                        if n == 'embeddings.{}.weight'.format(nat_idx_j) and nat_idx_j!=nat_idx:
                            connected &= dep
            if MASK_SCHEME == 1 or np.array_equal(ordering,
                                                  np.arange(num_cols)):
                assert ok
                assert connected

        print('[Transformer] Passes autoregressive-ness check!')
