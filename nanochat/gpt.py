"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768
    recurrent_layer_state: bool = False # enable recurrent layer state passing
    num_recurrence_warmup: int = 1 # number of warmup passes for training
    recurrent_layer_state_active: bool = True # allow compiling RLS but running in baseline mode
    # Ablation flags for testing RLS side token dominance hypothesis
    mask_side_attention: bool = False # prevent attention to side tokens during training
    zero_prev_state: bool = False # zero out prev_state to disable side stream
    side_dropout_rate: float = 0.15 # dropout rate for side stream (0.15 = 15% dropout, 1.0 = 100% dropout)
    side_type_gate: bool = False # scale side type embedding based on prev_state strength
    side_type_gate_temp: float = 4.0 # temperature for sigmoid gate
    side_type_gate_eps: float = 1e-6 # avoid division by zero when normalizing
    side_type_gate_ema_beta: float = 0.01 # update rate for running prev_state norm baseline
    side_stream_initial_scale: float = 0.1 # scheduled scale for prev_state/type embeddings at step 0
    side_stream_final_scale: float = 1.0 # scale after schedule completes
    side_stream_schedule_steps: int = 500 # number of steps to reach final scale
    side_type_scale: float = 1.0 # scalar multiplier for type embeddings (set <<1 to tame magnitude)
    side_type_renorm: bool = False # optionally RMSNorm after adding type embeddings
    side_state_rmsnorm: bool = False # RMSNorm prev_state before injecting into layer 0
    side_output_gate: bool = False # learnable sigmoid gate (per KV head) on side stream
    side_output_gate_init: float = -5.0 # initialization for the side gate logits
    side_logit_bias: float = 0.0 # additive bias applied to side logits before softmax
    side_logit_bias_trainable: bool = False # whether the side logit bias is learnable
    side_dual_softmax: bool = False # use separate softmaxes for main vs side streams


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config  # Store config for ablation flags
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.side_gate = None
        self.side_logit_bias = None
        if layer_idx == 0 and config.recurrent_layer_state:
            if config.side_output_gate:
                gate_init = torch.full(
                    (self.n_kv_head,),
                    config.side_output_gate_init,
                    dtype=torch.float32,
                )
                self.side_gate = nn.Parameter(gate_init)
            if config.side_logit_bias_trainable or config.side_logit_bias != 0.0:
                bias_init = torch.tensor(
                    config.side_logit_bias,
                    dtype=torch.float32,
                )
                self.side_logit_bias = nn.Parameter(bias_init)
                if not config.side_logit_bias_trainable:
                    self.side_logit_bias.requires_grad_(False)

    def forward(self, x, cos_sin, kv_cache, prev_state=None, rls_components=None):
        B, T, C = x.size()

        # Layer 0 with RLS: dual-stream attention (main tokens + side tokens from prev_state)
        if self.layer_idx == 0 and prev_state is not None and rls_components is not None:
            # Unpack RLS components
            if len(rls_components) == 4:
                E_type_main, E_type_side, type_gate, side_scale = rls_components
            elif len(rls_components) == 3:
                E_type_main, E_type_side, type_gate = rls_components
                side_scale = 1.0
            else:
                E_type_main, E_type_side = rls_components
                type_gate = None
                side_scale = 1.0

            # Add type embeddings to distinguish main vs side streams
            type_scale = float(getattr(self.config, "side_type_scale", 1.0))
            h_main = x
            if E_type_main is not None and type_scale != 0.0:
                type_vec = (type_scale * E_type_main).view(1, 1, -1).to(x.dtype)
                h_main = h_main + type_vec
            if getattr(self.config, "side_type_renorm", False):
                h_main = norm(h_main)

            scale_tensor = torch.as_tensor(side_scale, dtype=prev_state.dtype, device=prev_state.device)
            side_body = prev_state
            if getattr(self.config, "side_state_rmsnorm", False):
                side_body = norm(side_body)
            side_body = side_body * scale_tensor

            side_type = None
            if E_type_side is not None and type_scale != 0.0:
                base_type = (type_scale * E_type_side).view(1, 1, -1).to(prev_state.dtype)
                if type_gate is not None:
                    base_type = type_gate * base_type
                side_type = base_type

            if side_type is not None:
                h_side = side_body + side_type
            else:
                h_side = side_body
            if getattr(self.config, "side_type_renorm", False):
                h_side = norm(h_side)

            # Project queries (only from main stream - side tokens never query)
            q = self.c_q(h_main).view(B, T, self.n_head, self.head_dim)

            # Project keys and values for BOTH streams
            k_main = self.c_k(h_main).view(B, T, self.n_kv_head, self.head_dim)
            v_main = self.c_v(h_main).view(B, T, self.n_kv_head, self.head_dim)
            k_side = self.c_k(h_side).view(B, T, self.n_kv_head, self.head_dim)
            v_side = self.c_v(h_side).view(B, T, self.n_kv_head, self.head_dim)
            if self.side_gate is not None:
                gate = torch.sigmoid(self.side_gate)
                gate = gate.to(dtype=k_side.dtype, device=k_side.device).view(1, 1, self.n_kv_head, 1)
                k_side = k_side * gate
                v_side = v_side * gate

            # Apply RoPE to q and both k streams (same position indices)
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k_main = apply_rotary_emb(k_main, cos, sin)
            k_side = apply_rotary_emb(k_side, cos, sin)

            # QK norm
            q, k_main, k_side = norm(q), norm(k_main), norm(k_side)

            # Transpose to (B, H, T, D)
            q = q.transpose(1, 2)
            k_main = k_main.transpose(1, 2)
            v_main = v_main.transpose(1, 2)
            k_side = k_side.transpose(1, 2)
            v_side = v_side.transpose(1, 2)

            dual_softmax = getattr(self.config, "side_dual_softmax", False)

            # Concatenate main and side K/V along sequence dimension for caching
            k_combined = torch.cat([k_main, k_side], dim=2)  # (B, H, 2T, D)
            v_combined = torch.cat([v_main, v_side], dim=2)  # (B, H, 2T, D)

            # Handle KV cache for layer 0 (stores 2T positions)
            if kv_cache is not None:
                k_combined, v_combined = kv_cache.insert_kv(self.layer_idx, k_combined, v_combined)

            Tq = q.size(2)  # number of queries
            Tk_total = k_combined.size(2)  # total K/V length (2T for dual-stream)
            Tk_main = Tk_total // 2
            k_main_full = k_combined[:, :, :Tk_main, :]
            k_side_full = k_combined[:, :, Tk_main:, :]
            v_main_full = v_combined[:, :, :Tk_main, :]
            v_side_full = v_combined[:, :, Tk_main:, :]

            enable_gqa = self.n_head != self.n_kv_head

            def build_mask(stream_len, allow_attend=True):
                if not allow_attend:
                    return torch.zeros((Tq, stream_len), dtype=torch.bool, device=q.device)
                if kv_cache is None or Tq == stream_len:
                    return torch.tril(torch.ones((Tq, stream_len), dtype=torch.bool, device=q.device))
                if Tq == 1:
                    return torch.ones((1, stream_len), dtype=torch.bool, device=q.device)
                mask = torch.zeros((Tq, stream_len), dtype=torch.bool, device=q.device)
                prefix_len = max(stream_len - Tq, 0)
                if prefix_len > 0:
                    mask[:, :prefix_len] = True
                causal_width = stream_len - prefix_len
                if causal_width > 0:
                    mask[:, prefix_len:] = torch.tril(torch.ones((Tq, causal_width), dtype=torch.bool, device=q.device))
                return mask

            def build_bias(mask_bool, extra_bias=None):
                attn_bias = torch.full(mask_bool.shape, float("-inf"), dtype=q.dtype, device=q.device)
                attn_bias = attn_bias.masked_fill(mask_bool, 0.0)
                if extra_bias is not None:
                    attn_bias = attn_bias + extra_bias
                return attn_bias

            if dual_softmax:
                allow_side = not (self.training and self.config.mask_side_attention)
                mask_main = build_mask(Tk_main, True)
                mask_side = build_mask(Tk_main, allow_side)
                attn_bias_main = build_bias(mask_main)
                y_main = F.scaled_dot_product_attention(
                    q, k_main_full, v_main_full, attn_mask=attn_bias_main, enable_gqa=enable_gqa
                )
                has_side = allow_side and bool(mask_side.any().item())
                if has_side:
                    bias_val = None
                    if self.side_logit_bias is not None:
                        bias_val = self.side_logit_bias.to(dtype=q.dtype, device=q.device)
                    attn_bias_side = build_bias(mask_side, bias_val)
                    y_side = F.scaled_dot_product_attention(
                        q, k_side_full, v_side_full, attn_mask=attn_bias_side, enable_gqa=enable_gqa
                    )
                else:
                    y_side = torch.zeros_like(y_main)
                y = y_main + y_side
            else:
                k = k_combined
                v = v_combined

                def build_attn_bias(mask_main_bool, mask_side_bool):
                    allow = torch.cat([mask_main_bool, mask_side_bool], dim=1)
                    attn_bias = torch.full(
                        allow.shape,
                        float("-inf"),
                        dtype=q.dtype,
                        device=q.device,
                    )
                    attn_bias = attn_bias.masked_fill(allow, 0.0)
                    if self.side_logit_bias is not None:
                        bias_val = self.side_logit_bias.to(dtype=q.dtype, device=q.device)
                        attn_bias[:, Tk_main:] = attn_bias[:, Tk_main:] + bias_val
                    return attn_bias

                if kv_cache is None or Tq == Tk_main:
                    mask_main = torch.tril(torch.ones(Tq, Tk_main, dtype=torch.bool, device=q.device))
                    mask_side = torch.tril(torch.ones(Tq, Tk_main, dtype=torch.bool, device=q.device))
                    if self.training and self.config.mask_side_attention:
                        mask_side = torch.zeros_like(mask_side)
                    attn_bias = build_attn_bias(mask_main, mask_side)
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, enable_gqa=enable_gqa)
                elif Tq == 1:
                    mask_main = torch.ones(1, Tk_main, dtype=torch.bool, device=q.device)
                    mask_side = mask_main if not (self.training and self.config.mask_side_attention) else torch.zeros_like(mask_main)
                    attn_bias = build_attn_bias(mask_main, mask_side)
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False, enable_gqa=enable_gqa)
                else:
                    prefix_len = max(Tk_main - Tq, 0)
                    causal_width = Tk_main - prefix_len
                    mask_main = torch.zeros((Tq, Tk_main), dtype=torch.bool, device=q.device)
                    if prefix_len > 0:
                        mask_main[:, :prefix_len] = True
                    if causal_width > 0:
                        causal_block = torch.tril(torch.ones((Tq, causal_width), dtype=torch.bool, device=q.device))
                        mask_main[:, prefix_len:prefix_len + causal_width] = causal_block

                    if self.training and self.config.mask_side_attention:
                        mask_side = torch.zeros((Tq, Tk_main), dtype=torch.bool, device=q.device)
                    else:
                        mask_side = torch.zeros((Tq, Tk_main), dtype=torch.bool, device=q.device)
                        if prefix_len > 0:
                            mask_side[:, :prefix_len] = True
                        if causal_width > 0:
                            mask_side[:, prefix_len:prefix_len + causal_width] = causal_block
                    attn_bias = build_attn_bias(mask_main, mask_side)
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, enable_gqa=enable_gqa)

        else:
            # Normal single-stream attention (higher layers or no prev_state)
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

            # Apply Rotary Embeddings to queries and keys
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)  # QK norm
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # Apply KV cache
            if kv_cache is not None:
                k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            Tq = q.size(2)
            Tk = k.size(2)

            # Standard causal attention
            enable_gqa = self.n_head != self.n_kv_head
            if kv_cache is None or Tq == Tk:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
            elif Tq == 1:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
            else:
                attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
                prefix_len = Tk - Tq
                if prefix_len > 0:
                    attn_mask[:, :prefix_len] = True
                attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, Tq, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, prev_state=None, rls_components=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, prev_state, rls_components)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.recurrence_active = config.recurrent_layer_state and config.recurrent_layer_state_active
        self._last_type_gate_stats = None
        self._last_side_stream_enabled = False
        self._last_side_gate_stats = None
        self._last_side_logit_bias = None
        self.register_buffer("type_gate_ema", torch.tensor(1.0), persistent=False)
        # Recurrent layer state: side token architecture
        if config.recurrent_layer_state:
            # Type embeddings to distinguish main vs side token streams
            self.E_type_main = nn.Parameter(torch.zeros(config.n_embd))
            self.E_type_side = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.E_type_main = None
            self.E_type_side = None
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # Reinitialize side-specific parameters that lose data during to_empty()
            if getattr(block.attn, "side_gate", None) is not None:
                block.attn.side_gate.data.fill_(self.config.side_output_gate_init)
            if getattr(block.attn, "side_logit_bias", None) is not None:
                block.attn.side_logit_bias.data.fill_(self.config.side_logit_bias)
        # Initialize side MLP (already handled by _init_weights for Linear layers)
        # Type embeddings initialized to zeros (already done in __init__)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def set_recurrence_active(self, active: bool):
        """Toggle execution of the recurrent pathway without re-instantiating the model."""
        self.recurrence_active = bool(active) and self.config.recurrent_layer_state

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups (matrix, embedding, lm_head, RLS components)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # Type embeddings are 1D vectors, so they go to AdamW (Muon requires 2D+ matrices)
        rls_embedding_params = []
        if self.config.recurrent_layer_state:
            if self.E_type_main is not None:
                rls_embedding_params.append(self.E_type_main)
            if self.E_type_side is not None:
                rls_embedding_params.append(self.E_type_side)
            aux_params = []
            for block in self.transformer.h:
                side_gate = getattr(block.attn, "side_gate", None)
                if side_gate is not None:
                    aux_params.append(side_gate)
                bias_param = getattr(block.attn, "side_logit_bias", None)
                if bias_param is not None and bias_param.requires_grad:
                    aux_params.append(bias_param)
            for param in aux_params:
                if param in matrix_params:
                    matrix_params.remove(param)
            rls_embedding_params.extend(aux_params)
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(rls_embedding_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        # Add RLS type embeddings to AdamW with same LR as regular embeddings
        if self.config.recurrent_layer_state:
            adam_groups.append(dict(params=rls_embedding_params, lr=embedding_lr * dmodel_lr_scale))
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', prev_state=None, return_state=False, side_stream_scale=1.0):
        B, T = idx.size()
        recurrence_active = self.config.recurrent_layer_state and self.recurrence_active

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)

        # prev_state will be integrated via side tokens in layer 0's attention (if RLS enabled)
        # No mixing at the input layer - embeddings remain pure

        self._last_type_gate_stats = None
        self._last_side_stream_enabled = False
        self._last_side_gate_stats = None
        self._last_side_logit_bias = None
        if not recurrence_active:
            prev_state = None

        # ABLATION: Zero out prev_state to test if side token dominance causes failure
        if self.config.zero_prev_state and prev_state is not None and recurrence_active:
            prev_state = torch.zeros_like(prev_state)

        # Side stream dropout: occasionally run batches without RLS entirely to keep raw competence.
        # Removing the side stream (instead of zeroing it) avoids pushing gigantic gradients through identical tokens.
        side_stream_scale = float(side_stream_scale)

        if self.training and prev_state is not None and recurrence_active:
            if self.config.side_dropout_rate >= 1.0:
                prev_state = None
            elif self.config.side_dropout_rate > 0.0:
                if torch.rand((), device=prev_state.device) < self.config.side_dropout_rate:
                    prev_state = None

        # Optional type gating: scale E_type_side contribution based on prev_state strength
        type_gate = None
        if recurrence_active and prev_state is not None and self.config.side_type_gate:
            prev_norm = prev_state.norm(dim=-1, keepdim=True)
            ema = self.type_gate_ema.to(prev_state.device, prev_state.dtype)
            denom = ema + self.config.side_type_gate_eps
            ratio = prev_norm / denom
            gate = torch.sigmoid(self.config.side_type_gate_temp * (ratio - 1.0))
            gate_detached = gate.detach()
            self._last_type_gate_stats = (
                gate_detached.mean().item(),
                gate_detached.max().item(),
                gate_detached.min().item(),
            )
            type_gate = gate
            with torch.no_grad():
                ema_update = (1 - self.config.side_type_gate_ema_beta) * ema + self.config.side_type_gate_ema_beta * prev_norm.mean()
                self.type_gate_ema = ema_update.detach()

        # Pack RLS components for layer 0 (if enabled)
        rls_components = None
        if recurrence_active and prev_state is not None:
            rls_components = (self.E_type_main, self.E_type_side, type_gate, side_stream_scale)
            self._last_side_stream_enabled = True
        if self.config.recurrent_layer_state and self.transformer.h:
            attn0 = self.transformer.h[0].attn
            side_gate_param = getattr(attn0, "side_gate", None)
            if side_gate_param is not None:
                gate_vals = torch.sigmoid(side_gate_param.detach().float())
                if gate_vals.numel() > 0:
                    gate_mean = gate_vals.mean().item()
                    gate_max = gate_vals.max().item()
                    gate_min = gate_vals.min().item()
                    self._last_side_gate_stats = (gate_mean, gate_max, gate_min)
            bias_param = getattr(attn0, "side_logit_bias", None)
            if bias_param is not None:
                self._last_side_logit_bias = float(bias_param.detach().item())

        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, prev_state=prev_state, rls_components=rls_components)
        x = norm(x)

        # Capture final state before lm_head
        final_state = x

        # If we only need state (warmup mode), skip expensive lm_head computation
        # Don't skip if kv_cache is provided (generation mode needs logits)
        if return_state and targets is None and kv_cache is None:
            return None, final_state

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            # Note: Keeping logits in bfloat16 instead of converting to float32
            # to save memory. PyTorch cross_entropy handles bfloat16 fine.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_state:
                return loss, final_state
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            if return_state:
                return logits, final_state
            return logits

    def forward_with_recurrence(self, idx, targets=None, loss_reduction='mean', side_stream_scale=1.0):
        """
        Forward pass with recurrent layer state.
        Performs warmup passes to compute previous token states, then does the real forward pass.
        """
        recurrence_active = self.config.recurrent_layer_state and self.recurrence_active
        if not recurrence_active:
            # Feature disabled, use normal forward
            return self.forward(idx, targets=targets, loss_reduction=loss_reduction)

        B, T = idx.size()
        device = idx.device
        # Infer dtype from model parameters (bfloat16 on CUDA, float32 on CPU)
        model_dtype = next(self.parameters()).dtype

        # Start with zeros for previous state
        prev_state = torch.zeros(B, T, self.config.n_embd, dtype=model_dtype, device=device)

        # Perform warmup passes (no gradients)
        # Note: forward() will skip lm_head when targets=None and return_state=True
        for i in range(self.config.num_recurrence_warmup):
            with torch.no_grad():
                _, warmup_state = self.forward(idx, targets=None, prev_state=prev_state, return_state=True, side_stream_scale=side_stream_scale)
                # Shift: position i gets position i-1's output, position 0 gets zeros
                prev_state = torch.cat([
                    torch.zeros(B, 1, self.config.n_embd, dtype=model_dtype, device=device),
                    warmup_state[:, :-1, :]
                ], dim=1)

        # Real forward pass with gradients
        return self.forward(idx, targets=targets, loss_reduction=loss_reduction, prev_state=prev_state, return_state=False, side_stream_scale=side_stream_scale)

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        recurrence_active = self.config.recurrent_layer_state and self.recurrence_active
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Initial setup
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        prev_state = None

        # If using recurrent layer state, do warmup passes on the initial prompt
        if recurrence_active:
            T = ids.size(1)
            prev_state = torch.zeros(1, T, self.config.n_embd, dtype=torch.bfloat16, device=device)
            for _ in range(self.config.num_recurrence_warmup):
                _, warmup_state = self.forward(ids, prev_state=prev_state, return_state=True)
                # Shift: position i gets position i-1's output
                prev_state = torch.cat([
                    torch.zeros(1, 1, self.config.n_embd, dtype=torch.bfloat16, device=device),
                    warmup_state[:, :-1, :]
                ], dim=1)
            # Get the last token's state for generation
            _, final_state = self.forward(ids, prev_state=prev_state, return_state=True)
            prev_state = final_state[:, -1:, :]  # (1, 1, n_embd)

        for _ in range(max_tokens):
            if recurrence_active:
                # Generate next token with recurrent state
                next_id = torch.tensor([[ids[0, -1].item()]], dtype=torch.long, device=device)  # Last token as (1, 1)
                logits, final_state = self.forward(next_id, prev_state=prev_state, return_state=True)  # (1, 1, vocab_size)
                logits = logits[:, -1, :]  # (1, vocab_size)
                prev_state = final_state  # Update for next iteration
            else:
                logits = self.forward(ids) # (B, T, vocab_size)
                logits = logits[:, -1, :] # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
