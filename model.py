import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data


class Bert(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)
        self.sigma_map = None
        if config.cond_dim is not None:
            self.sigma_map = TimestepEmbedding(config.cond_dim)
            print("Using sigma map for conditional embeddings")

    def get_contextualized(self, input_ids, attention_mask, mask_p):
        if self.sigma_map is not None:
            t_cond = F.silu(self.sigma_map(mask_p))
        else:
            t_cond = None
        static_embeddings, relative_embedding = self.embedding(input_ids)
        contextualized_embeddings, intermid_outputs = self.transformer(static_embeddings, attention_mask.unsqueeze(1), relative_embedding, t_cond)
        return contextualized_embeddings, intermid_outputs

    def forward(self, input_ids, attention_mask, masked_lm_labels, ratio=None, mask_p=None, sum=False):        
        contextualized_embeddings, intermid_outputs = self.get_contextualized(input_ids, attention_mask, mask_p)
        subword_prediction = self.classifier(contextualized_embeddings, masked_lm_labels)
        gold_labels = masked_lm_labels.flatten()
        gold_labels = gold_labels[gold_labels != -100]


        if sum:
            valid_mask = (masked_lm_labels != -100)
            seq_len, batch_size = masked_lm_labels.shape
            seq_ids = torch.arange(batch_size, device=masked_lm_labels.device).unsqueeze(0).expand(seq_len, -1)
            sequence_ids = seq_ids[valid_mask]
            # Compute per-token loss
            loss_per_token = F.cross_entropy(subword_prediction, gold_labels, reduction="none")
            loss = torch.zeros(batch_size, device=loss_per_token.device)
            loss.scatter_add_(0, sequence_ids, loss_per_token)

        else:
            gold_labels = masked_lm_labels.flatten()
            gold_labels = gold_labels[gold_labels != -100]
            loss = F.cross_entropy(subword_prediction, gold_labels, reduction="none").mean()
        with torch.no_grad():
            accuracy = (subword_prediction.argmax(-1) == gold_labels).float().mean()
        num_tokens = gold_labels.size(0)
        return loss, accuracy, num_tokens


# From https://github.com/epfml/DenseFormer
class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.Tensor().to(full_tensor.device)
        ret.set_(full_tensor[:x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice



class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.attention_layers = nn.ModuleList([Attention(config) for _ in range(config.num_hidden_layers)])
        self.mlp_layers = nn.ModuleList([FeedForward(config) for _ in range(config.num_hidden_layers)])
        
        for i, layer in enumerate(self.mlp_layers):
            layer.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
        
        self.activation_checkpointing = activation_checkpointing
        
        # One adaLN_modulation per layer for timestep conditioning
        if config.cond_dim is not None:
            self.adaLN_modulations = nn.ModuleList([
                nn.Linear(config.cond_dim, 6 * config.hidden_size) 
                for _ in range(config.num_hidden_layers)
            ])
            
            # Zero initialize all adaLN modulation layers
            for modulation in self.adaLN_modulations:
                modulation.weight.data.zero_()
                modulation.bias.data.zero_()
        else:
            self.adaLN_modulations = None  # No modulation if cond_dim is None

    def forward(self, x, attention_mask, relative_embedding, t_cond=None):
        intermid_outputs = None
                
        for i, (attention_layer, mlp_layer) in enumerate(zip(self.attention_layers, self.mlp_layers)):
            if t_cond is not None:
                # Get timestep modulation parameters for this layer
                (shift_msa, scale_msa, gate_msa, shift_mlp, 
                 scale_mlp, gate_mlp) = self.adaLN_modulations[i](t_cond)[None, :].chunk(6, dim=2)
            else:
                # No timestep conditioning - use None values and identity gating
                shift_msa = scale_msa = shift_mlp = scale_mlp = None
                gate_msa = gate_mlp = torch.ones(1, device=x.device, dtype=x.dtype)
            
            # Store skip connection for attention
            x_skip = x
            
            # Attention with timestep modulation (pass modulation to attention layer)
            attn_out = attention_layer(x, attention_mask, relative_embedding, 
                                     shift_msa, scale_msa)
            x = x_skip + gate_msa * attn_out  # Apply gating
            
            x_skip = x
            
            # MLP with timestep modulation (pass modulation to mlp layer)
            mlp_out = mlp_layer(x, shift_mlp, scale_mlp)
            x = x_skip + gate_mlp * mlp_out  # Apply gating            
                
        return x, intermid_outputs

@torch.jit.script
def gather_nonpad_ts(x: torch.Tensor, labels: torch.Tensor):
    x_flat      = x.reshape(-1, x.size(-1))
    labels_flat = labels.reshape(-1)
    mask        = labels_flat != -100
    return x_flat[mask]

class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels):
        x = gather_nonpad_ts(x, masked_lm_labels)
        x = self.nonlinearity(x)
        return x

class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x



class FeedForward(nn.Module):
    def __init__(self, config): 
        super().__init__()
        
        # Keep the mlp attribute for backward compatibility with Encoder initialization
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        self.layer_norm1 = self.mlp[0]
        self.linear1 = self.mlp[1] 
        self.geglu = self.mlp[2]
        self.layer_norm2 = self.mlp[3]
        self.linear2 = self.mlp[4]
        self.dropout = self.mlp[5]
        
        self.initialize(config.hidden_size)
    
    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)
    
    def forward(self, x, shift=None, scale=None):
        # Apply first layer norm
        x = self.layer_norm1(x)
        
        # Apply timestep modulation if provided
        if shift is not None and scale is not None:
            x = modulate(x, shift, scale)
        
        # Continue with the rest of the feedforward computation
        x = self.linear1(x)
        x = self.geglu(x)
        x = self.layer_norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x
class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_vg = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_vg.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_vg.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, hidden_states, attention_mask, relative_embedding, shift=None, scale=None):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)

        if shift is not None and scale is not None:
            hidden_states = modulate(hidden_states, shift, scale)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value, gate = self.in_proj_vg(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        gate = F.gelu(gate)

        pos = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)  # shape: [T, T, 2D]
        query_pos, key_pos = pos.chunk(2, dim=-1)
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.head_size)
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.head_size)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = context * gate
        context = self.post_layer_norm(context)
        context = self.out_proj(context)
        context = self.dropout(context)

        return context


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

class TimestepEmbedding(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb