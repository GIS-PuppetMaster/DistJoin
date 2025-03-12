"""MADE and ResMADE."""
import copy
import typing
from typing import List

from ordered_set import OrderedSet
from sympy import factor

import common
import distributions
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util
from utils.util import get_device, InvertOrder, MakeIMDBTables


def seqList_2_EnsembleSeq(models: List[torch.nn.Sequential]) -> torch.nn.Sequential:
    num_model = len(models)
    device = list(models[0].parameters())[0].data.device
    # [model, layer]
    models_ = np.array(list(map(lambda m: list(m), models)), dtype=object)
    layers = np.array(list(map(lambda m: len(m), models_)))
    num_layer = layers[0]
    ens_layers = []
    assert (layers == num_layer).all()
    sep_input_sizes = None
    ens_input_size = None
    for i in range(num_layer):
        layers = models_[:, i]
        layer_types = np.array(list(map(lambda l: type(l), layers)))
        layer_type = layer_types[0]
        assert (layer_types == layer_type).all()
        if layer_type == torch.nn.Linear:
            in_features = np.array(list(map(lambda l: l.in_features, layers)))
            if i == 0:
                assert sep_input_sizes is None and ens_input_size is None
                sep_input_sizes = in_features
                ens_input_size = np.sum(in_features).item()
            out_features = np.array(list(map(lambda l: l.out_features, layers)))
            ens_layer = EnsembleLinear(np.sum(in_features).item(), np.sum(out_features).item())
            ens_layer.set_model(layers)
            ens_layers.append(ens_layer)
        else:
            if hasattr(layers[0], 'inplace'):
                ens_layers.append(type(layers[0])(inplace=True))
            else:
                ens_layers.append(type(layers[0])())
    ens_model = torch.nn.Sequential(*ens_layers).to(device)
    # test
    # sep_inputs = []
    # sep_outputs = []
    # for i in range(num_model):
    #     sep_inputs.append(torch.rand(1, sep_input_sizes[i], device=device))
    #     sep_outputs.append(models[i](sep_inputs[-1]))
    # sep_output = torch.concat(sep_outputs, dim=-1)
    # ens_input = torch.concat(sep_inputs, dim=-1)
    # ens_output = ens_model(ens_input)
    # assert torch.abs((sep_output - ens_output) < 1e-6).all()
    return ens_model


class EnsembleLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('block_gradient_mask',
                             torch.zeros_like(self.weight.data, device=self.weight.data.device, requires_grad=False))

    def set_model(self, layers: List[torch.nn.Linear]):
        self.weight.data.zero_()
        self.bias.data.zero_()
        x = 0
        y = 0
        z = 0
        for layer in layers:
            w = layer.weight.data
            b = layer.bias.data
            self.weight.data[x:x + w.shape[0], y:y + w.shape[1]] = w
            self.block_gradient_mask[x:x + w.shape[0], y:y + w.shape[1]] = 1.
            x += w.shape[0]
            y += w.shape[1]
            self.bias.data[z:z + b.shape[0]] = b
            z += b.shape[0]


class MaskedLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 condition_on_ordering=False):
        super().__init__(in_features, out_features, bias)

        self.register_buffer('mask', torch.ones(out_features, in_features, requires_grad=False))

        self.condition_ordering_linear = None
        if condition_on_ordering:
            self.condition_ordering_linear = nn.Linear(in_features,
                                                       out_features,
                                                       bias=False)

        self.masked_weight = None

    def set_mask(self, mask):
        """Accepts a mask of shape [in_features, out_features]."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def set_cached_mask(self, mask):
        self.mask.data.copy_(mask)

    def get_cached_mask(self):
        return self.mask.clone().detach()

    def forward(self, input):
        if self.masked_weight is None:
            return F.linear(input, self.mask * self.weight, self.bias)
        else:
            # ~17% speedup for Prog Sampling.
            out = F.linear(input, self.masked_weight, self.bias)

        if self.condition_ordering_linear is None:
            return out
        return out + F.linear(torch.ones_like(input),
                              self.mask * self.condition_ordering_linear.weight)


class MaskedResidualBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 condition_on_ordering=False,
                 resmade_drop_prob=0.):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.dropout = nn.Dropout(p=resmade_drop_prob)
        self.activation = activation

    def set_mask(self, mask):
        self.layers[0].set_mask(mask)
        self.layers[1].set_mask(mask)

    def set_cached_mask(self, mask):
        # They have the same mask.
        self.layers[0].mask.copy_(mask)
        self.layers[1].mask.copy_(mask)

    def get_cached_mask(self):
        return self.layers[0].mask.clone().detach()

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layers[1](out)
        return input + out


class ActivationWarper(nn.Module):
    def __init__(self, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(x)


def swish(x):
    return x * torch.sigmoid(x)


class MixedActivation(nn.Module):
    def __init__(self, num_features=None):
        super().__init__()
        self.activations = nn.ModuleList([
            nn.Identity(),
            nn.ReLU(),
            nn.PReLU(),
            nn.LeakyReLU(),
            nn.ELU(),
            nn.GELU(),
            nn.SELU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.SiLU(),
            ActivationWarper(torch.sin),
            ActivationWarper(torch.cos),
            ActivationWarper(torch.arctan)
        ])
        self.num_features = num_features
        self.num_act = len(self.activations)
        if self.num_features is None:
            # [activations, 1(bs), 1(feature)]
            self.weights = nn.Parameter(torch.ones(self.num_act, 1, 1))
        else:
            # [feature, activations]
            self.weights = nn.Parameter(torch.ones(num_features, self.num_act))

    def forward(self, x):
        z = torch.stack([act(x) for act in self.activations], dim=0)
        if self.num_features is None:
            return torch.sum(z * self.weights, dim=0) / torch.sum(self.weights)
        else:
            assert False, "this leaks information and breaks the autoregressive"
            # [bs, features] * [features, activations] -> [bs, activations]
            weights = torch.matmul(x, self.weights)
            # [bs, activations] -> [activations, bs] -> [activations, bs, 1]
            # [activations, bs, features] * [activations, bs, 1] -> [activations, bs, features] -> [bs, features]
            return torch.sum(z * weights.transpose(0, 1).unsqueeze(-1), dim=0) / torch.sum(weights)


class MADE(nn.Module):
    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=1,
            factor_table=None,
            natural_ordering=True,
            input_bins=None,
            activation=nn.ReLU,
            do_direct_io_connections=False,
            input_encoding=None,
            output_encoding='one_hot',
            embed_size=32,
            input_no_emb_if_leq=True,
            embs_tied=True,
            residual_connections=False,
            seed=11123,
            fixed_ordering=None,
            # Wildcard skipping.
            dropout_p=True,
            fixed_dropout_p=False,
            learnable_unk=None,
            grouped_dropout=False,
            per_row_dropout=False,
            # DMoL.
            num_dmol=0,
            scale_input=False,
            dmol_col_indexes=[],
            resmade_drop_prob=0.,
            multi_pred_embedding='mlp',
            use_ensemble=False,
            use_adaptive_temp=True,
            use_ANPM=True,
            use_sos=True,
            use_mix_act=True
    ):
        """MADE and ResMADE.

        Args:
          nin: integer; number of input variables.  Each input variable
            represents a column.
          hidden_sizes: a list of integers; number of units in hidden layers.
          nout: integer; number of outputs, the sum of all input variables'
            domain sizes.
          num_masks: number of orderings + connectivity masks to cycle through.
          natural_ordering: force natural ordering of dimensions, don't use
            random permutations.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          activation: the activation to use.
          do_direct_io_connections: whether to add a connection from inputs to
            output layer.  Helpful for information flow.
          input_encoding: input encoding mode, see EncodeInput().
          output_encoding: output logits decoding mode, either 'embed' or
            'one_hot'.  See logits_for_col().
          embed_size: int, embedding dim.
          input_no_emb_if_leq: optimization, whether to turn off embedding for
            variables that have a domain size less than embed_size.  If so,
            those variables would have no learnable embeddings and instead are
            encoded as one hot vecs.
          residual_connections: use ResMADE?  Could lead to faster learning.
            Recommended to be set for any non-trivial datasets.
          seed: seed for generating random connectivity masks.
          fixed_ordering: variable ordering to use.  If specified, order[i]
            maps natural index i -> position in ordering.  E.g., if order[0] =
            2, variable 0 is placed at position 2.
          dropout_p, learnable_unk: if True, turn on column masking during
            training time, which enables the wildcard skipping (variable
            skipping) optimization during inference.  Recommended to be set for
            any non-trivial datasets.
          grouped_dropout: bool, whether to mask factorized subvars for an
            original var together or independently.
          per_row_dropout: bool, whether to make masking decisions per tuple or
            per batch.
          num_dmol, scale_input, dmol_col_indexes: (experimental) use
            discretized mixture of logistics as outputs for certain columns.
          num_joined_tables: int, number of joined tables.
          resmade_drop_prob: float, normal dropout probability inside ResMADE.
        """
        super().__init__()
        print('fixed_ordering', fixed_ordering, 'seed', seed,
              'natural_ordering', natural_ordering)

        # self.original_input_bins = [c.distribution_size for c in factor_table.base_table.columns]
        # # replace last_fact_column size to original column size
        # last_key_fact_id = factor_table.ori_id_to_fact_id[factor_table.key_col_ori_idx][-1]
        # for last_fact_idx in factor_table.last_facts_idx:
        #     idx = factor_table.fact_id_to_ori_id[last_fact_idx]
        #     if last_fact_idx!= last_key_fact_id:
        #         input_bins[last_fact_idx] = factor_table.base_table.columns[idx].distribution_size
        self.use_sos = use_sos
        self.use_adp_temp = use_adaptive_temp
        if self.use_sos:
            # sos
            input_bins.insert(0, embed_size)
            nin += 1

        self.nin = nin

        assert multi_pred_embedding in ['cat', 'mlp', 'rnn', 'rec']
        assert input_encoding in [None, 'one_hot', 'binary', 'embed']
        if num_masks > 1:
            # Double the weights, so need to reduce the size to be fair.
            hidden_sizes = [int(h // 2 ** 0.5) for h in hidden_sizes]
            print('Auto reducing MO hidden sizes to', hidden_sizes, num_masks)
        # None: feed inputs as-is, no encoding applied.  Each column thus
        #     occupies 1 slot in the input layer.  For testing only.
        self.use_ANPM = use_ANPM
        self.use_mix_act = use_mix_act
        self.input_encoding = input_encoding
        assert output_encoding in ['one_hot', 'embed']
        if num_dmol > 0:
            assert output_encoding == 'embed'
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        self.input_no_emb_if_leq = input_no_emb_if_leq
        self.do_direct_io_connections = do_direct_io_connections
        assert not embs_tied
        self.embs_tied = embs_tied
        self.factor_table = factor_table
        self.multi_pred_embedding = multi_pred_embedding

        self.residual_connections = residual_connections

        self.num_masks = num_masks
        self.learnable_unk = learnable_unk
        self.dropout_p = dropout_p
        self.fixed_dropout_p = fixed_dropout_p
        self.grouped_dropout = grouped_dropout
        self.per_row_dropout = per_row_dropout
        if self.per_row_dropout:
            assert self.dropout_p

        self.resmade_drop_prob = resmade_drop_prob

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1
            print('** Fixed ordering {} supplied, ignoring natural_ordering'.
                  format(fixed_ordering))

        # Discretized MoL.
        self.num_dmol = num_dmol
        self.scale_input = scale_input
        self.dmol_col_indexes = dmol_col_indexes

        self.unk_embedding_pred_cache = None
        assert self.input_bins
        encoded_bins, self.input_bins_encoded, self.input_bins_encoded_cumsum, self.input_bins_encoded_raw, self.input_bins_encoded_raw_cumsum, self.input_bins_encoded_no_pred_cumsum = self.SetBins(
            self.input_bins, self.nin)
        self.out_bins = encoded_bins
        print('encoded_bins (output)', encoded_bins)
        print('encoded_bins (input)', self.input_bins_encoded)

        hs = [nin] + hidden_sizes + [sum(encoded_bins)]
        self.net = []
        for i, (h0, h1) in enumerate(zip(hs, hs[1:])):
            if residual_connections:
                if i == 0 or i == len(hs) - 2:
                    if i == len(hs) - 2:
                        self.net.append(MixedActivation() if use_mix_act else self.activation(inplace=True))
                    # Input / Output layer.
                    self.net.extend([
                        MaskedLinear(h0,
                                     h1,
                                     condition_on_ordering=self.num_masks > 1)
                    ])
                else:
                    # Middle residual blocks must have same dims.
                    assert h0 == h1, (h0, h1, hs)
                    self.net.extend([
                        MaskedResidualBlock(
                            h0,
                            h1,
                            activation=self.activation(inplace=True),
                            condition_on_ordering=self.num_masks > 1,
                            resmade_drop_prob=self.resmade_drop_prob)
                    ])
            else:
                self.net.extend([
                    MaskedLinear(h0,
                                 h1,
                                 condition_on_ordering=self.num_masks > 1),
                    MixedActivation() if use_mix_act else self.activation(inplace=True),
                ])
        if not residual_connections:
            self.net.pop()
        self.net = nn.Sequential(*self.net)

        self.first_col_prob_head = None

        # Input layer should be changed.
        assert self.input_bins is not None
        input_size = 0
        for i, dist_size in enumerate(self.input_bins):
            input_size += self._get_input_encoded_dist_size(dist_size, i)
        new_layer0 = MaskedLinear(input_size,
                                  self.net[0].out_features,
                                  condition_on_ordering=self.num_masks > 1)
        self.net[0] = new_layer0

        self.adaptive_temperature = nn.Parameter(torch.ones(self.nin, )) if use_adaptive_temp else None

        # 生成条件嵌入调制网络
        modulation_embeds = [None, ] # None for SOS
        modulation_offset_layers_to_hidden = [None, ]
        modulation_offset_layers_to_hidden2 = [None, ]
        modulation_offset_layers_hidden_bias = [None, ]
        modulation_offset_layers_to_logits = [None, ]
        modulation_offset_layers_to_logits2 = [None, ]
        modulation_offset_layers_logits_bias = [None, ]
        for ori_idx, ori_col in enumerate(self.factor_table.base_table.columns):
            for j, fact_idx in enumerate(self.factor_table.ori_id_to_fact_id[ori_idx]):
                modulation_embeds.append(nn.Embedding(self.input_bins[fact_idx + (1 if self.use_sos else 0)], self.embed_size))

                if j == 0 or not self.use_ANPM:
                    modulation_offset_layers_to_hidden.append(None)
                    modulation_offset_layers_to_hidden2.append(None)
                    modulation_offset_layers_hidden_bias.append(None)
                    modulation_offset_layers_to_logits.append(None)
                    modulation_offset_layers_to_logits2.append(None)
                    modulation_offset_layers_logits_bias.append(None)
                else:
                    modulation_offset_layers_to_hidden.append(nn.Sequential(*[
                        nn.Linear(self.embed_size*j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.embed_size),
                    ]))
                    modulation_offset_layers_to_hidden2.append(nn.Sequential(*[
                        nn.Linear(self.embed_size * j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.out_bins[fact_idx + (1 if self.use_sos else 0)]),
                    ]))
                    modulation_offset_layers_hidden_bias.append(nn.Sequential(*[
                        nn.Linear(self.embed_size * j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.embed_size),
                    ]))
                    modulation_offset_layers_to_logits.append(nn.Sequential(*[
                        nn.Linear(self.embed_size * j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.embed_size),
                    ]))
                    modulation_offset_layers_to_logits2.append(nn.Sequential(*[
                        nn.Linear(self.embed_size * j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.out_bins[fact_idx + (1 if self.use_sos else 0)]),
                    ]))
                    modulation_offset_layers_logits_bias.append(nn.Sequential(*[
                        nn.Linear(self.embed_size * j, self.embed_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_size, self.out_bins[fact_idx + (1 if self.use_sos else 0)]),
                    ]))
        self.modulation_embeds = nn.ModuleList(modulation_embeds)
        self.modulation_offset_layers_to_hidden = nn.ModuleList(modulation_offset_layers_to_hidden)
        self.modulation_offset_layers_to_hidden2 = nn.ModuleList(modulation_offset_layers_to_hidden2)
        self.modulation_offset_layers_hidden_bias = nn.ModuleList(modulation_offset_layers_hidden_bias)
        self.modulation_offset_layers_to_logits = nn.ModuleList(modulation_offset_layers_to_logits)
        self.modulation_offset_layers_to_logits2 = nn.ModuleList(modulation_offset_layers_to_logits2)
        self.modulation_offset_layers_logits_bias = nn.ModuleList(modulation_offset_layers_logits_bias)

        if self.output_encoding == 'embed':
            assert self.input_encoding == 'embed'
        if self.input_encoding == 'embed':
            self.embeddings = nn.ModuleList()
            self.embeddings_out = nn.ModuleList()
            for i, dist_size in enumerate(self.input_bins):
                if dist_size < self.embed_size and self.input_no_emb_if_leq:
                    embed = None
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                self.embeddings.append(embed)
            for i, dist_size in enumerate(self.input_bins):
                if dist_size < self.embed_size and self.input_no_emb_if_leq:
                    embed = None
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                self.embeddings_out.append(embed)

        # Learnable [MASK] representation.
        assert self.dropout_p  # not allow to disable wild-card skipping due to performance reason
        self.unk_embeddings = nn.ParameterList()
        for i, dist_size in enumerate(self.input_bins):
            self.unk_embeddings.append(
                nn.Parameter(torch.randn(1, self.input_bins_encoded[i] - 5)))
        self.use_ensemble = use_ensemble
        self.multi_pred_embed_nn = None
        self.is_ensembled = False

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.output_bins_cumsum = self.logit_indices
        self.m = {}
        # self.cached_masks = {}

        self.update_masks()
        self.orderings = [self.m[-1]]
        self.nat_order = util.InvertOrder(self.m[-1])
        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = [None] * self.nin
        const_one = torch.ones([], dtype=torch.long, device=get_device())
        for i, coli_dom_size in enumerate(self.input_bins):
            # Max with 1 to guard against cols with 1 distinct val.
            one_hot_dims = max(1, int(np.ceil(np.log2(coli_dom_size))))
            self.bin_as_onehot_shifts[i] = const_one << torch.arange(
                one_hot_dims, device=get_device())

    def ConvertToEnsemble(self):
        if self.multi_pred_embedding == 'mlp':
            if self.use_ensemble and not self.is_ensembled:
                self.multi_pred_embed_nn_ = self.multi_pred_embed_nn
                self.multi_pred_embed_nn = seqList_2_EnsembleSeq(list(self.multi_pred_embed_nn))
                self.is_ensembled = True
                print('using ensemble')

    def ConvertToUnensemble(self):
        if self.is_ensembled:
            self.multi_pred_embed_nn = self.multi_pred_embed_nn_
            del self.multi_pred_embed_nn_
            self.is_ensembled = False
            print('using multiple models')

    def SetBins(self, input_bins, nin):
        col_idxs = list(range(len(input_bins)))
        encoded_bins = list(
            map(self._get_output_encoded_dist_size, input_bins, col_idxs))
        input_bins_encoded = list(
            map(self._get_input_encoded_dist_size, input_bins, col_idxs))
        input_bins_encoded_cumsum = np.cumsum(
            list(map(self._get_input_encoded_dist_size, input_bins, col_idxs)))
        input_bins_encoded_raw = [
            self._get_input_encoded_dist_size(input_bins[i], i) - 5
            for i in range(len(input_bins))
        ]
        input_bins_encoded_raw_cumsum = np.cumsum(input_bins_encoded_raw)
        input_bins_encoded_no_pred_cumsum = (input_bins_encoded_cumsum - np.ones(nin) * 5).astype(int)
        return encoded_bins, input_bins_encoded, input_bins_encoded_cumsum, input_bins_encoded_raw, input_bins_encoded_raw_cumsum, input_bins_encoded_no_pred_cumsum

    def UseDMoL(self, natural_idx):
        """Returns True if we want to use DMoL for this column."""
        if self.num_dmol <= 0:
            return False
        return natural_idx in self.dmol_col_indexes

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(
                direct_nin,
                direct_nout,
                condition_on_ordering=self.num_masks > 1)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        # Inverse: ord_idx -> natural idx.
        inv_ordering = InvertOrder(self.m[-1])

        for ord_i in range(self.nin):
            nat_i = inv_ordering[ord_i]
            if self.use_sos:
                ori_nat_i = self.factor_table.fact_id_to_ori_id[int(nat_i) - 1] + 1 if nat_i > 0 else 0
            else:
                ori_nat_i = self.factor_table.fact_id_to_ori_id[int(nat_i)]
            # x_(nat_i) in the input occupies range [inp_l, inp_r).
            inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[nat_i - 1]
            inp_r = self.input_bins_encoded_cumsum[nat_i]
            assert inp_l < inp_r

            for ord_j in range(ord_i + 1, self.nin):
                nat_j = inv_ordering[ord_j]
                if self.use_sos:
                    ori_nat_j = self.factor_table.fact_id_to_ori_id[int(nat_j) - 1] + 1 if nat_j > 0 else 0
                else:
                    ori_nat_j = self.factor_table.fact_id_to_ori_id[int(nat_j)]
                if self.use_ANPM and ori_nat_i == ori_nat_j:
                    continue
                # Output x_(nat_j) should connect to input x_(nat_i); it
                # occupies range [out_l, out_r) in the output.
                out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                out_r = self.logit_indices[nat_j]
                assert out_l < out_r
                mask[out_l:out_r, inp_l:inp_r] = 1
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, dist_size, i):
        del i  # Unused.
        if self.input_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.input_encoding == 'one_hot':
            pass
        elif self.input_encoding == 'binary':
            dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        elif self.input_encoding is None:
            dist_size = 1
        else:
            assert False, self.input_encoding
        dist_size += 5
        if self.multi_pred_embedding == 'cat':
            dist_size *= 2
        return dist_size

    def _get_output_encoded_dist_size(self, dist_size, i):
        if self.output_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.UseDMoL(i):
            dist_size = self.num_dmol * 3
        elif self.output_encoding == 'one_hot':
            pass
        return dist_size

    def update_masks(self, invoke_order=None, force=False):
        """Update m() for all layers and change masks correspondingly.

        This implements multi-order training support.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1 and not force:
            return
        L = len(self.hidden_sizes)

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
                                   isinstance(l, MaskedResidualBlock)
        ]

        ### Precedence of several params determining ordering:
        #
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        #
        # from high precedence to low.

        # For multi-order models, we associate RNG seeds with orderings as
        # follows:
        #   orderings = [ o0, o1, o2, ... ]
        #   seeds = [ 0, 1, 2, ... ]
        # This must be consistent across training & inference.

        if invoke_order is not None:
            assert False, 'deprecated'

        elif hasattr(self, 'orderings'):
            # Training path: cycle through the special orderings.
            assert False, 'deprecated'

        else:
            # Train-time initial construction: either single-order, or
            # .orderings has not been assigned yet.
            rng = np.random.RandomState(self.seed)
            print('creating rng with seed', self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(
                self.nin - (1 if self.use_sos else 0)) if self.natural_ordering else rng.permutation(
                self.nin - (1 if self.use_sos else 0))
            if self.fixed_ordering is not None:
                assert len(self.fixed_ordering) == self.nin
                self.m[-1] = np.asarray(self.fixed_ordering)

        if self.factor_table is not None:
            # Correct for subvar ordering.
            # This could have [..., 6, ..., 4, ..., 5, ...].
            # So we map them back into:
            # This could have [..., 4, 5, 6, ...].
            # Subvars have to be in order and also consecutive
            order = self.m[-1]
            # order_idx -> nat_order
            order = util.InvertOrder(order)
            for orig_col, sub_cols in self.factor_table.map_from_col_to_fact_col.items():
                first_subvar_index = self.factor_table.columns_name_to_idx[sub_cols[0]]

                print('Before', order)
                if len(sub_cols) > 1:
                    for j in range(1, len(sub_cols)):
                        subvar_index = self.factor_table.columns_name_to_idx[sub_cols[j]]
                        order = np.delete(order,
                                          np.argwhere(order == subvar_index))
                        order = np.insert(
                            order,
                            np.argwhere(order == first_subvar_index)[0][0] + j,
                            subvar_index)
                    print('After', order)
            print('Inverted Special order', order)

            # 把key列放最后
            import AQP_estimator
            key_col_name = AQP_estimator.JoinOrderBenchmark.GetJobLightJoinKeys()[self.factor_table.base_table.name]
            key_fact_col_names = self.factor_table.map_from_col_to_fact_col[key_col_name]
            # 不允许分解key列
            # assert len(key_fact_col_name) == 1 and key_col_name == key_fact_col_name[0], str(key_col_name) + str(key_fact_col_name)
            for key_fact_col_name in key_fact_col_names:
                key_fact_col_idx = self.factor_table.columns_name_to_idx[key_fact_col_name]
                order = np.delete(order, np.argwhere(order == key_fact_col_idx))
                order = np.insert(order, len(order), key_fact_col_idx)
                print(f'Order after move key column: {key_fact_col_name}({key_fact_col_idx}) to last: {order}')
            # nat_order->order_idx
            self.m[-1] = util.InvertOrder(order)
        if self.use_sos:
            self.m[-1] = self.m[-1] + 1
            self.m[-1] = np.insert(self.m[-1], 0, 0)  # insert SOS

        print('self.seed', self.seed, 'self.m[-1]', self.m[-1])

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    assert len(np.unique(
                        self.hidden_sizes)) == 1, self.hidden_sizes
                    # Sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    if l > 0:
                        self.m[l] = self.m[0]
                    else:
                        # ar = np.arange(self.nin-1)
                        # self.m[l] = rng.choice(ar, self.hidden_sizes[l], replace=True, p=(ar+1)/(ar+1).sum())
                        self.m[l] = np.array([
                            (k - 1) % (self.nin - 1)
                            for k in range(self.hidden_sizes[l])
                        ])
                        if self.num_masks > 1:
                            self.m[l] = rng.permutation(self.m[l])
                else:
                    # Samples from [0, ncols - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        causal_masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        causal_masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # 自定义mask
        fact_id_to_ori_id = {0: 0}
        for k, v in self.factor_table.fact_id_to_ori_id.items():
            if self.use_sos:
                fact_id_to_ori_id[k + 1] = v + 1  # right shift
            else:
                fact_id_to_ori_id[k] = v

        fact_col_m = {}
        ori_col_m = {}
        ori_col_idx_ord_m = {}

        idx_ord_to_nat_ord = {}
        for ord_id, nat_idx in enumerate(util.InvertOrder(self.m[-1])):
            idx_ord_to_nat_ord[ord_id] = int(nat_idx)

        ori_nat_order_ = OrderedSet()
        for fact_nat_idx in util.InvertOrder(self.m[-1]):
            if self.use_sos:
                ori_nat_idx = fact_id_to_ori_id[int(fact_nat_idx) - 1] + 1 if fact_nat_idx > 0 else 0
            else:
                ori_nat_idx = fact_id_to_ori_id[int(fact_nat_idx)]
            ori_nat_order_.add(ori_nat_idx)
        ori_nat_order_ = list(ori_nat_order_)

        for l, m_l in self.m.items():
            fact_col_m[l] = np.array([idx_ord_to_nat_ord[ord_idx] for ord_idx in m_l])  # ord_idx -> nat_idx # util.fast_get_idx(m_l, self.m[-1])
            # last_fact_col_m[l] = np.isin(fact_col_m[l], last_fact_idx)
            ori_col_m[l] = np.array([fact_id_to_ori_id[x] if x > 0 else 0 for x in fact_col_m[l]])
            ori_col_idx_ord_m[l] = np.array([ori_nat_order_.index(x) for x in ori_col_m[l]])

        same_fact_col_masks = [fact_col_m[l - 1][:, None] == fact_col_m[l][None, :] for l in range(L)]
        same_fact_col_masks.append(fact_col_m[L - 1][:, None] == fact_col_m[-1][None, :])

        same_ori_col_masks = [ori_col_m[l - 1][:, None] == ori_col_m[l][None, :] for l in range(L)]
        same_ori_col_masks.append(ori_col_m[L - 1][:, None] == ori_col_m[-1][None, :])

        prev_ori_col_masks = [ori_col_idx_ord_m[l - 1][:, None] < ori_col_idx_ord_m[l][None, :] for l in range(L)]
        prev_ori_col_masks.append(ori_col_idx_ord_m[L - 1][:, None] < ori_col_idx_ord_m[-1][None, :])

        masks = []
        for l in range(len(causal_masks)):
            mask = causal_masks[l]
            if l < L:
                # 中间层的掩码
                # 对于tgt（输出）中所有last_fact_col的，只能与自己和前驱col的last_fact_col连接
                # 对于tgt（输出）中所有非last_fact_col的，只能与自己、同col的前驱fact_col以及前驱col的last_fact_col连接
                mask &= same_fact_col_masks[l] | same_ori_col_masks[l] | prev_ori_col_masks[l]
            else:  # l==L
                # 最后一层的掩码，只能与 非自己&(前驱fact_col,前驱col的last_fact_col) 连接
                if self.use_ANPM:
                    mask &= (~same_fact_col_masks[l]) & (~same_ori_col_masks[l]) & prev_ori_col_masks[l]
                else:
                    mask &= (~same_fact_col_masks[l]) & prev_ori_col_masks[l]
            masks.append(mask)

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # Replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(
                            x[1], idx)
                        tmp_mask.extend([mval] * nbins)
                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        # Input layer's mask should be changed.

        assert self.input_bins is not None
        # [nin, hidden].
        mask0 = masks[0]
        new_mask0 = []
        for i, dist_size in enumerate(self.input_bins):
            dist_size = self._get_input_encoded_dist_size(dist_size, i)
            # [dist size, hidden]
            new_mask0.append(
                np.concatenate([mask0[i].reshape(1, -1)] * dist_size,
                               axis=0))
        # [sum(dist size), hidden]
        new_mask0 = np.vstack(new_mask0)
        masks[0] = new_mask0

        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        if self.do_direct_io_connections:
            self._build_or_update_direct_io()

    def name(self):
        n = 'made'
        if self.residual_connections:
            n += '-resmade'
        n += '-hidden' + '_'.join(str(h) for h in self.hidden_sizes)
        n += '-emb' + str(self.embed_size)
        if self.num_masks > 1:
            n += '-{}masks'.format(self.num_masks)
        if not self.natural_ordering:
            n += '-nonNatural'
        n += ('-no' if not self.do_direct_io_connections else '-') + 'directIo'
        n += '-{}In{}Out'.format(self.input_encoding, self.output_encoding)
        n += '-embsTied' if self.embs_tied else '-embsNotTied'
        if self.input_no_emb_if_leq:
            n += '-inputNoEmbIfLeq'
        n += '-' + self.multi_pred_embedding
        if self.num_dmol > 0:
            n += '-DMoL{}'.format(self.num_dmol)
            if self.scale_input:
                n += '-scale'
        if self.dropout_p:
            n += '-dropout'
            if self.learnable_unk:
                n += '-learnableUnk'
            if self.fixed_dropout_p:
                n += '-fixedDropout{:.2f}'.format(self.dropout_p)
        if self.factor_table:
            n += '-factorized'
            if self.dropout_p and self.grouped_dropout:
                n += '-groupedDropout'
            n += '-{}wsb'.format(str(self.factor_table.word_size_bits))
        return n

    def Embed(self, data, pred, natural_col=None, out=None, merge=True, force_not_use_wildcard=False):
        input_bins = self.input_bins
        # assert data is not None

        bs = data.size()[0]
        y_embed = [None] * len(input_bins)
        data = data.long()

        for i, coli_dom_size in enumerate(input_bins):
            # # Wildcard column? use -1 as special token.
            # # Inference pass only (see estimators.py).
            # skip = data[0][i] < 0
            data_i = data.narrow(-1, i, 1)
            none_pred_mask = (data_i == -1).to(torch.float32, non_blocking=True, copy=False)
            # 把不存在的val从-1替换为0，因为有none_pred_mask记录其位置，后续不会产生错误结果，便于embedding计算
            data_i = torch.where(data_i == -1, 0, data_i)
            dropped_repr = self.unk_embeddings[i].to(data.device)
            # Embed?
            if coli_dom_size >= self.embed_size or not self.input_no_emb_if_leq:
                col_i_embs = self.embeddings[i](data_i.squeeze(-1))
                pred_i = pred.narrow(-1, i * 5, 5)
                y_embed[i] = col_i_embs
                y_embed[i] = dropped_repr * none_pred_mask + y_embed[i] * (1.0 - none_pred_mask)
            else:
                y_onehot = torch.zeros(bs,
                                       data_i.shape[1],
                                       coli_dom_size,
                                       device=data.device)
                y_onehot = torch.scatter(y_onehot, -1, data_i, 1)
                pred_i = pred.narrow(-1, i * 5, 5)
                y_embed[i] = y_onehot
                y_embed[i] = dropped_repr * none_pred_mask + y_embed[i] * (1.0 - none_pred_mask)
            y_embed[i] = torch.cat([y_embed[i], pred_i], dim=-1)
        if merge:
            return torch.cat(y_embed, -1, out=out)
        else:
            if out is not None:
                out.copy_(y_embed)
            return y_embed

    def EncodeInput(self, data, preds, natural_col=None, out=None, train=False):
        """Encodes token IDs.

        Warning: this could take up a significant portion of a forward pass.

        Args:
          data: torch.Long, shaped [N, 2, nin] or [N, 1].
          preds: torch.Long, shaped [N, 2, 5*nin]
          natural_col: if specified, 'data' has shape [N, 1] corresponding to
              col-'natural-col'.  Otherwise 'data' corresponds to all cols.
          out: if specified, assign results into this Tensor storage.
        """
        # add SOS for x
        if self.use_sos:
            data = torch.cat([torch.full((data.shape[0], data.shape[1], 1), -1, dtype=torch.long, device=data.device), data], dim=-1)
            preds = torch.cat([torch.zeros((preds.shape[0], preds.shape[1], 5), device=preds.device), preds], dim=-1)
        # data, preds = util.sortPredicates(data, preds)
        if self.input_encoding == 'binary':
            assert False, 'deprecated'
        elif self.input_encoding == 'embed':
            return self.Embed(data, preds, natural_col=natural_col, out=out, merge=True)
        elif self.input_encoding == 'onehot':
            assert False, 'deprecated'
        elif self.input_encoding is None:
            assert False, 'deprecated'
        else:
            assert False, self.input_encoding

    def merge_multi_preds(self, x):
        # x is a list of length self.nin, each x[i].shape == (batch, 2, encode_size)
        return torch.sum(x, dim=1)

    def forward(self, x):
        """Calculates unnormalized logits.

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x: [bs, ncols].
        """
        # x is a list consists of tensor (bs, preds, encode) for each column
        vals, preds = x[0], x[1]
        x = self.EncodeInput(vals, preds)
        return self.forward_with_encoded_input(x)

    def forward_with_encoded_input(self, x):
        x = self.merge_multi_preds(x)
        # x = self.mapping_factorized_preds(x)
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def logits_for_col(self, idx, logits, pred_vs=None):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering, includes SOS.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          pred_vs: input predicates' values, not includes SOS

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        if idx == 0:
            logits_for_var = logits.narrow(-1, 0, int(self.logit_indices[0]))
        else:
            logits_for_var = logits.narrow(-1, int(self.logit_indices[idx - 1]),
                                           int(self.logit_indices[idx]) - int(self.logit_indices[idx - 1]))

        T = self.adaptive_temperature[idx] if self.adaptive_temperature is not None else 1

        if self.use_sos:
            ori_idx = self.factor_table.fact_id_to_ori_id[idx - 1] if idx != 0 else 0
        else:
            ori_idx = self.factor_table.fact_id_to_ori_id[idx]
        if (offset_net_to_hidden:=self.modulation_offset_layers_to_hidden[idx]) is not None:
            offset_net_to_hidden2 = self.modulation_offset_layers_to_hidden2[idx]
            offset_net_hidden_bias = self.modulation_offset_layers_hidden_bias[idx]
            offset_net_to_logits = self.modulation_offset_layers_to_logits[idx]
            offset_net_to_logits2 = self.modulation_offset_layers_to_logits2[idx]
            offset_net_logits_bias = self.modulation_offset_layers_logits_bias[idx]
            fact_idxs = self.factor_table.ori_id_to_fact_id[ori_idx]
            assert len(fact_idxs)>1
            if self.training:
                value_embed = []
                for fact_idx in fact_idxs:
                    if fact_idx>=idx-(1 if self.use_sos else 0):
                        break
                    embed = self.modulation_embeds[fact_idx+(1 if self.use_sos else 0)]
                    pred_v = pred_vs[..., fact_idx].long()
                    # invalid = pred_v == -1
                    # pred_v = torch.where(invalid, 0, pred_v)
                    # invalid = invalid.float().unsqueeze(-1)
                    # value_embed.append((embed(pred_v) * (1. - invalid)).sum(dim=1))
                    value_embed.append(embed(pred_v))
                value_embed = torch.cat(value_embed, dim=-1)
            else:
                value_embed = None
                for fact_idx in fact_idxs:
                    if fact_idx>=idx-(1 if self.use_sos else 0):
                        break
                    embed = self.modulation_embeds[fact_idx+(1 if self.use_sos else 0)]
                    if value_embed is None:
                        value_embed = embed.weight
                    else:
                        # 扩展第一个嵌入矩阵到三维 (N1, N2, F1)
                        emb1 = value_embed
                        emb2 = embed.weight
                        expanded_emb1 = emb1.unsqueeze(1)  # (N1, 1, F1)
                        expanded_emb1 = expanded_emb1.expand(-1, emb2.size(0), -1)  # (N1, N2, F1)

                        # 扩展第二个嵌入矩阵到三维 (N1, N2, F2)
                        expanded_emb2 = emb2.unsqueeze(0)  # (1, N2, F2)
                        expanded_emb2 = expanded_emb2.expand(emb1.size(0), -1, -1)  # (N1, N2, F2)

                        # 拼接特征维度并展平
                        value_embed = torch.cat([expanded_emb1, expanded_emb2], dim=-1)  # (N1, N2, F1+F2)
                        value_embed = value_embed.view(-1, value_embed.size(-1))  # (N1*N2, F1+F2)
            to_hidden = offset_net_to_hidden(value_embed)
            to_hidden2 = offset_net_to_hidden2(value_embed)
            hidden_bias = offset_net_hidden_bias(value_embed)
            to_logits = offset_net_to_logits(value_embed)
            to_logits2 = offset_net_to_logits2(value_embed)
            logits_bias = offset_net_logits_bias(value_embed)
            to_hidden_w = torch.bmm(to_hidden2.unsqueeze(-1), to_hidden.unsqueeze(1))
            to_logits_w = torch.bmm(to_logits.unsqueeze(-1), to_logits2.unsqueeze(1))
            if logits_for_var.shape[0] != to_logits_w.shape[0]:
                logits_for_var = logits_for_var.expand(to_hidden_w.size(0), -1)
            logits_for_var = torch.nn.functional.relu(
                torch.bmm(torch.nn.functional.relu(
                    torch.bmm(logits_for_var.unsqueeze(1), to_hidden_w) + hidden_bias.unsqueeze(1))
                                                                , to_logits_w)+logits_bias.unsqueeze(1)).squeeze(1) # training: (bs, f), eval: (\prod{NDV_higher}, f)

        if self.output_encoding != 'embed':
            return logits_for_var / T

        if self.embs_tied:
            embed = self.embeddings[idx]
        else:
            embed = self.embeddings_out[idx]

        if embed is None:
            # Can be None for small-domain columns.
            return logits_for_var / T

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        t = embed.weight.t()
        # Inference path will pass in output buffer, which shaves off a bit of
        # latency.
        return torch.matmul(logits_for_var, t) / T

    def nll(self, logits, data, pred_vs, table: typing.Union[common.TableDataset, common.FactorizedTable], label_smoothing=0, use_weight=False):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          data: [batch size, nin].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(1 if self.use_sos else 0, self.nin):  # skip SOS output
            logits_i = self.logits_for_col(i, logits, pred_vs)
            loss = F.cross_entropy(logits_i,
                                   data[:, i - (1 if self.use_sos else 0)],  # data has not been padding sos
                                   reduction='none',
                                   weight=table.columns[i - (1 if self.use_sos else 0)].dist_weights if use_weight else None)
            assert loss.size() == nll.size()

            nll += loss  # * weight
        return nll


if __name__ == '__main__':
    from colorama import Fore, Back, Style, init

    # Checks for the autoregressive property.
    rng = np.random.RandomState(14)
    # (nin, hiddens, nout, input_bins, direct_io, residual_connections)
    configs_with_input_bins = [
        (True, [], False, False),
        (True, [128], False, False),
        (True, [128, 256], False, False),
        (True, [128], True, False),
        (True, [128, 256], True, False),
        (True, [256, 256], False, True),
        (True, [256, 256], True, True),
        (True, [256, 256, 256, 256], True, True),

        (False, [], False, False),
        (False, [128], False, False),
        (False, [128, 256], False, False),
        (False, [128], True, False),
        (False, [128, 256], True, False),
        (False, [256, 256], False, True),
        (False, [256, 256], True, True),
        (False, [256, 256, 256, 256], True, True),
    ]
    table = MakeIMDBTables('title', data_dir='../datasets/job/', use_cols='general')
    factor_table = common.FactorizedTable(common.TableDataset(table, bs=1),
                                          word_size_bits=12,
                                          factorize_blacklist=[])
    nin = len(factor_table.columns)
    input_bins = [c.DistributionSize() for c in factor_table.columns]
    nout = sum(input_bins)
    for natural_ordering, hiddens, direct_io, res in configs_with_input_bins:
        print(natural_ordering, hiddens, direct_io, res, '...', end='')
        model = MADE(nin,
                     hiddens,
                     nout,
                     factor_table=factor_table,
                     input_encoding='embed',
                     output_encoding='embed',
                     input_bins=copy.copy(input_bins),
                     natural_ordering=natural_ordering,
                     do_direct_io_connections=direct_io,
                     residual_connections=res,
                     embs_tied=False,
                     )
        model.eval()
        # print(model)

        idx_order = model.m[-1]
        nat_order = util.InvertOrder(idx_order)
        ori_nat_order = list(OrderedSet([model.factor_table.fact_id_to_ori_id[int(i) - 1] + 1 if int(i) > 0 else 0 for i in nat_order]))
        ori_idx_order = util.InvertOrder(ori_nat_order)
        print(f'ori_nat_order: {ori_nat_order}')
        for i_out in nat_order:
            inp = []
            for i in range(1, model.nin):
                inp.append(rng.randint(0, model.input_bins[i], size=(1, 2, 1)))
            inp = torch.tensor(np.concatenate(inp, axis=-1).astype(float), requires_grad=True)
            preds = torch.tile(torch.tensor([[1., 0., 0., 0., 0.] * nin], requires_grad=True).unsqueeze(1),
                               dims=(1, 2, 1))
            x = model.EncodeInput(inp, preds, train=False)
            x.retain_grad()
            logits = model.forward_with_encoded_input(x)[-1]

            model.logits_for_col(i_out, logits, inp, reduce=False).sum().backward(retain_graph=True)
            ori_id_i_out = model.factor_table.fact_id_to_ori_id[i_out - 1] + 1 if i_out > 0 else 0
            grad = x.grad.detach().clone()
            for j_in in range(model.nin):
                grad_j = grad[..., :model.input_bins_encoded_cumsum[0]] if j_in == 0 else grad[..., model.input_bins_encoded_cumsum[j_in - 1]:model.input_bins_encoded_cumsum[j_in]]
                ori_id_j_in = model.factor_table.fact_id_to_ori_id[j_in - 1] + 1 if j_in > 0 else 0
                # j_in_is_last = model.factor_table.last_facts_idx[ori_id_j_in] == j_in
                allowed = False
                # if idx_order[j_in] < idx_order[i_out]:
                #     allowed = True
                if ori_idx_order[ori_id_j_in] < ori_idx_order[ori_id_i_out]:  # and j_in_is_last:
                    allowed = True
                # if ori_id_j_in == ori_id_i_out and idx_order[j_in] < idx_order[i_out]:
                #     allowed = True
                if allowed and (grad_j == 0).any():
                    print(Fore.RED + f'Missing dependency between input: {ori_id_j_in}-{j_in} to output: {ori_id_i_out}-{i_out}')
                elif (not allowed) and (grad_j != 0).any():
                    print(Fore.RED + f'Illegal dependency between input: {ori_id_j_in}-{j_in} to output: {ori_id_i_out}-{i_out}')
                else:
                    tmp = 'allowed' if allowed else 'not allowed'
                    print(Fore.WHITE + f'Checked {tmp} dependency between input: {ori_id_j_in}-{j_in} to output: {ori_id_i_out}-{i_out}')
        print(Fore.LIGHTWHITE_EX + 'finished one pass')
    print(Fore.LIGHTWHITE_EX + 'finished all')
    # print('[MADE] Passes autoregressive-ness check!')
