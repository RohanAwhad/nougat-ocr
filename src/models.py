import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Iterable
from typing import Tuple, Optional

# Encoder
# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


class DonutSwinEmbeddings(nn.Module):
  def __init__(self, config):
    super().__init__()
    image_size, patch_size = config.image_size, config.patch_size
    num_channels, hidden_size = config.num_channels, config.embed_dim
    patch_size = (patch_size, patch_size)
    num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    self.image_size = image_size
    self.patch_size = patch_size
    self.num_channels = num_channels
    self.num_patches = num_patches
    self.patch_grid = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

    self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    self.mask_token = None
    self.position_embeddings = None

    self.norm = nn.LayerNorm(config.embed_dim)

  def maybe_pad(self, pixel_values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if width % self.patch_size[1] != 0:
      pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
      pixel_values = F.pad(pixel_values, pad_values)
    if height % self.patch_size[0] != 0:
      pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
      pixel_values = F.pad(pixel_values, pad_values)
    return pixel_values

  def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, num_channels, height, width = pixel_values.shape
    # pad the input to be divisible by self.patch_size, if needed
    pixel_values = self.maybe_pad(pixel_values, height, width)
    embeddings = self.projection(pixel_values)
    _, _, height, width = embeddings.shape
    output_dimensions = (height, width)
    embeddings = embeddings.flatten(2).transpose(1, 2)
    embeddings = self.norm(embeddings)
    batch_size, seq_len, _ = embeddings.size()

    return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class DonutSwinPatchMerging(nn.Module):
  """
  Patch Merging Layer.

  Args:
      input_resolution (`Tuple[int]`):
          Resolution of input feature.
      dim (`int`):
          Number of input channels.
      norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
          Normalization layer class.
  """

  def __init__(self, input_resolution: Tuple[int], dim: int) -> None:
    super().__init__()
    self.input_resolution = input_resolution
    self.dim = dim
    self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
    self.norm = nn.LayerNorm(4 * dim)

  def maybe_pad(self, input_feature, height, width):
    should_pad = (height % 2 == 1) or (width % 2 == 1)
    if should_pad:
        pad_values = (0, 0, 0, width % 2, 0, height % 2)
        input_feature = nn.functional.pad(input_feature, pad_values)

    return input_feature

  def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
    height, width = input_dimensions
    # `dim` is height * width
    batch_size, dim, num_channels = input_feature.shape

    input_feature = input_feature.view(batch_size, height, width, num_channels)
    # merge a 2x2 patch into a single vector
    # pad input to be disible by width and height, if needed
    input_feature = self.maybe_pad(input_feature, height, width)
    # [batch_size, height/2, width/2, num_channels]
    input_feature_0 = input_feature[:, 0::2, 0::2, :]
    # [batch_size, height/2, width/2, num_channels]
    input_feature_1 = input_feature[:, 1::2, 0::2, :]
    # [batch_size, height/2, width/2, num_channels]
    input_feature_2 = input_feature[:, 0::2, 1::2, :]
    # [batch_size, height/2, width/2, num_channels]
    input_feature_3 = input_feature[:, 1::2, 1::2, :]
    # batch_size height/2 width/2 4*num_channels
    input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
    input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

    input_feature = self.norm(input_feature)
    input_feature = self.reduction(input_feature)

    return input_feature


class DonutSwinAttention(nn.Module):
  def __init__(self, config, dim, num_heads, window_size):
    super().__init__()

    if dim % num_heads != 0:
      raise ValueError(
        f"embedding dimension = {dim} should be divisible by number of heads = {num_heads}"
      )

    self.num_attention_heads = num_heads
    self.attention_head_size = dim // num_heads
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.window_size = (
        window_size if isinstance(window_size, Iterable) else (window_size, window_size)
    )

    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
    )

    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(self.window_size[0])
    coords_w = torch.arange(self.window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += self.window_size[0] - 1
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    self.register_buffer("relative_position_index", relative_position_index)

    self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
    self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
    self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
    self.out_proj = nn.Linear(self.all_head_size, self.all_head_size)

  def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    bz, seq_len, embed_dim = hidden_states.shape
    
    q = self.query(hidden_states)
    k = self.key(hidden_states)
    v = self.value(hidden_states)

    q = q.view(bz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
    k = k.view(bz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
    v = v.view(bz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    attn_scores = q @ k.transpose(-1, -2)
    attn_scores = attn_scores / (self.attention_head_size ** 0.5)
    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
    relative_position_bias = relative_position_bias.view(
      self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
    )
    attn_scores += relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    if attention_mask is not None:
      # Apply the attention mask is (precomputed for all layers in DonutSwinModel forward() function)
      mask_shape = attention_mask.shape[0]
      attn_scores = attn_scores.view(
        bz // mask_shape, mask_shape, self.num_attention_heads, seq_len, seq_len
      )
      attn_scores = attn_scores + attention_mask.unsqueeze(1).unsqueeze(0)
    
    attn_scores = attn_scores.view(-1, self.num_attention_heads, seq_len, seq_len)
    attn_scores = F.softmax(attn_scores, dim=-1)

    context_layer = (attn_scores @ v).permute(0, 2, 1, 3).contiguous().view(bz, seq_len, embed_dim)
    out = self.out_proj(context_layer)
    return out


class DonutSwinFFN(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
    self.fc2 = nn.Linear(4*embed_dim, embed_dim)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.gelu(self.fc1(hidden_states))
    hidden_states = self.fc2(hidden_states)
    return hidden_states

# Copied from transformers.models.swin.modeling_swin.SwinLayer with Swin->DonutSwin
class DonutSwinLayer(nn.Module):
  def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
    super().__init__()
    self.chunk_size_feed_forward = config.chunk_size_feed_forward
    self.shift_size = shift_size
    self.window_size = config.window_size
    self.input_resolution = input_resolution  # this seems to be useless. But might be useful for compiling the graph
    self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
    self.attention = DonutSwinAttention(config, dim, num_heads, window_size=self.window_size)
    self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
    self.ffn = DonutSwinFFN(dim)


  def set_shift_and_window_size(self, input_resolution):
    if min(input_resolution) <= self.window_size:
      # if window size is larger than input resolution, we don't partition windows
      self.shift_size = 0
      self.window_size = min(input_resolution)

  def get_attn_mask(self, height, width, dtype):
    # TODO (rohan): remove conditional logic
    if self.shift_size > 0:
      # calculate attention mask for SW-MSA
      img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
      height_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      width_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      count = 0
      for height_slice in height_slices:
        for width_slice in width_slices:
          img_mask[:, height_slice, width_slice, :] = count
          count += 1

      mask_windows = window_partition(img_mask, self.window_size)
      mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
      attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
      attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        attn_mask = None
    return attn_mask

  def maybe_pad(self, hidden_states, height, width):
    pad_right = (self.window_size - width % self.window_size) % self.window_size
    pad_bottom = (self.window_size - height % self.window_size) % self.window_size
    pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
    hidden_states = F.pad(hidden_states, pad_values)
    return hidden_states, pad_values

  def forward(
    self,
    hidden_states: torch.Tensor,
    input_dimensions: Tuple[int, int]
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    # self.set_shift_and_window_size(input_dimensions)
    height, width = input_dimensions
    batch_size, _, channels = hidden_states.size()

    shortcut = hidden_states
    hidden_states = self.layernorm_before(hidden_states)

    hidden_states = hidden_states.view(batch_size, height, width, channels)
    # pad hidden_states to multiples of window size
    hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

    _, height_pad, width_pad, _ = hidden_states.shape
    # cyclic shift
    if self.shift_size > 0:
        shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_hidden_states = hidden_states

    # partition windows
    hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
    hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
    attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
    if attn_mask is not None: attn_mask = attn_mask.to(hidden_states_windows.device)

    attention_output = self.attention(hidden_states_windows, attn_mask)
    attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
    shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

    # reverse cyclic shift
    if self.shift_size > 0:
        attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        attention_windows = shifted_windows

    was_padded = pad_values[3] > 0 or pad_values[5] > 0
    if was_padded: attention_windows = attention_windows[:, :height, :width, :].contiguous()

    attention_windows = attention_windows.view(batch_size, height * width, channels)
    hidden_states = shortcut + attention_windows

    layer_output = self.layernorm_after(hidden_states)
    layer_output = hidden_states + self.ffn(layer_output)
    return layer_output


# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->DonutSwin
class DonutSwinStage(nn.Module):
  def __init__(self, config, dim, input_resolution, depth, num_heads, downsample):
      super().__init__()
      self.config = config
      self.dim = dim
      self.blocks = nn.ModuleList(
          [
              DonutSwinLayer(
                  config=config,
                  dim=dim,
                  input_resolution=input_resolution,
                  num_heads=num_heads,
                  shift_size=0 if (i % 2 == 0) else config.window_size // 2,
              )
              for i in range(depth)
          ]
      )

      # patch merging layer
      self.downsample = downsample(input_resolution, dim=dim) if downsample is not None else None

  def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
      height, width = input_dimensions
      for i, layer_module in enumerate(self.blocks):
        hidden_states = layer_module(hidden_states, input_dimensions)

      hidden_states_before_downsampling = hidden_states
      if self.downsample is not None:
          height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
          output_dimensions = (height, width, height_downsampled, width_downsampled)
          hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
      else:
          output_dimensions = (height, width, height, width)

      return (hidden_states, output_dimensions)


# Copied from transformers.models.swin.modeling_swin.SwinEncoder with Swin->DonutSwin
class DonutSwinEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embedder = DonutSwinEmbeddings(config)
    grid_size = self.embedder.patch_grid
    self.num_layers = len(config.depths)
    self.config = config
    self.layers = nn.ModuleList(
        [
            DonutSwinStage(
                config=config,
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                downsample=DonutSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            for i_layer in range(self.num_layers)
        ]
    )

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    hidden_states, input_dimensions = self.embedder(pixel_values)
    for i, layer_module in enumerate(self.layers):
      hidden_states, output_dimensions = layer_module(hidden_states, input_dimensions)
      input_dimensions = (output_dimensions[-2], output_dimensions[-1])

    return hidden_states


# Decoder (Auto Regressive Model)
class MBartSelfAttention(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, max_len: int, is_cross_attn: bool, is_causal: bool):
    super().__init__()

    if (embed_dim % num_heads) != 0:
      raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.key = nn.Linear(embed_dim, embed_dim)
    self.value = nn.Linear(embed_dim, embed_dim)
    self.query = nn.Linear(embed_dim, embed_dim)  # keeping it below key and value for aligning with saved state dict in transformer
    self.out_proj = nn.Linear(embed_dim, embed_dim)

    if is_cross_attn: self.attn = self.cross_attn
    else: self.attn = self.self_attn


    causal_mask = torch.zeros(max_len, max_len)
    if is_causal and max_len is not None:
      causal_mask = causal_mask.masked_fill(torch.tril(torch.ones(max_len, max_len)) == 0, float('-inf'))
    self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    self.is_causal = is_causal # TODO (rohan): delet this

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_states: Optional[torch.Tensor] = None
  ) -> torch.Tensor:

    bz, _, embed_dim = hidden_states.shape

    (q, k, v), (q_seq_len, kv_seq_len), (q_seq_begin, q_seq_end) = self.attn(hidden_states, kv_states, k_cache, v_cache)

    q = q.view(bz, q_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = k.view(bz, kv_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = v.view(bz, kv_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    attn_scores = q @ k.transpose(-1, -2)
    attn_scores = attn_scores / (self.head_dim ** 0.5)

    attn_scores += attention_mask + self.causal_mask[q_seq_begin:q_seq_end, :kv_seq_len]
    attn_scores = F.softmax(attn_scores, dim=-1)
    context_layer = (attn_scores @ v).permute(0, 2, 1, 3).contiguous().view(bz, q_seq_len, embed_dim)
    out = self.out_proj(context_layer)

    # reshape k, v
    k = k.permute(0, 2, 1, 3).contiguous().view(bz, kv_seq_len, embed_dim)
    v = v.permute(0, 2, 1, 3).contiguous().view(bz, kv_seq_len, embed_dim)
    return out, k, v

  def cross_attn(self, hidden_states: torch.Tensor, kv_states: torch.Tensor, k_cache, v_cache) -> Tuple[torch.Tensor]:
    q = self.query(hidden_states)
    if (k_cache.shape == kv_states.shape) and (v_cache.shape == kv_states.shape):
      k = k_cache
      v = v_cache
    else:
      k = self.key(kv_states)
      v = self.value(kv_states)

    q_seq_len, kv_seq_len = hidden_states.shape[1], kv_states.shape[1]
    return (q, k, v), (q_seq_len, kv_seq_len), (0, q_seq_len)

  def self_attn(self, hidden_states: torch.Tensor, kv_states, k_cache, v_cache) -> torch.Tensor:
    q = self.query(hidden_states)
    k = self.key(hidden_states)
    v = self.value(hidden_states)

    k = torch.cat([k_cache, k], dim=1)
    v = torch.cat([v_cache, v], dim=1)

    # I need q_seq_len
    q_seq_begin = k_cache.shape[1]
    q_seq_end = q_seq_begin + hidden_states.shape[1]
    q_seq_len, kv_seq_len = hidden_states.shape[1], k.shape[1]
    return (q, k, v), (q_seq_len, kv_seq_len), (q_seq_begin, q_seq_end)


class MBartFFN(nn.Module):
  def __init__(self, embed_dim: int):
    super().__init__()

    self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
    self.fc2 = nn.Linear(4*embed_dim, embed_dim)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.gelu(self.fc1(hidden_states))
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class MBartLayer(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, max_len: int):
    super().__init__()

    self.self_attn = MBartSelfAttention(embed_dim, num_heads, is_cross_attn=False, is_causal=True, max_len=max_len)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.cross_attn = MBartSelfAttention(embed_dim, num_heads, is_cross_attn=True, is_causal=False, max_len=max_len)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.ffn = MBartFFN(embed_dim)
    self.ln3 = nn.LayerNorm(embed_dim)

  def forward(
    self,
    hidden_states: torch.Tensor,
    kv_states: torch.Tensor,
    self_attention_mask: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    ks_cache: torch.Tensor,
    vs_cache: torch.Tensor,
    kc_cache: torch.Tensor,
    vc_cache: torch.Tensor,
  ) -> torch.Tensor:

    x = self.ln1(hidden_states)
    x, ks_cache, vs_cache = self.self_attn(x, self_attention_mask, ks_cache, vs_cache)
    hidden_states += x

    x = self.ln2(hidden_states)
    x, kc_cache, vc_cache = self.cross_attn(x, cross_attention_mask, kc_cache, vc_cache, kv_states)
    hidden_states += x

    x = self.ln3(hidden_states)
    hidden_states += self.ffn(x)

    return hidden_states, ks_cache, vs_cache, kc_cache, vc_cache

class MBartDecoder(nn.Module):
  def __init__(self, embed_dim: int, num_layers: int, num_heads: int, vocab_size: int, max_len: int, scale_embedding: bool = False):
    super().__init__()

    self.embed_scale = (embed_dim ** 0.5) if scale_embedding else 1.0
    self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
    self.offset = 2
    self.position_embeddings = nn.Embedding(max_len+self.offset, embed_dim)  # offset as in transformers implementation of MBart
    self.layers = nn.ModuleList([MBartLayer(embed_dim, num_heads, max_len=max_len) for _ in range(num_layers)])
    self.ln_begin = nn.LayerNorm(embed_dim)
    self.ln_end = nn.LayerNorm(embed_dim)

    self.register_buffer("position_ids", torch.arange(max_len).unsqueeze(0) + self.offset, persistent=False)

  def forward(
    self,
    input_ids: torch.Tensor,
    self_attention_mask: torch.Tensor,
    kv_states: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    ks_cache: torch.Tensor,  # (n_layers, bz, seq_len, embed_dim)
    vs_cache: torch.Tensor,  # (n_layers, bz, seq_len, embed_dim)
    kc_cache: torch.Tensor,  # (n_layers, bz, seq_len, encoder_embed_dim)
    vc_cache: torch.Tensor,  # (n_layers, bz, seq_len, encoder_embed_dim)
    s_cache_pos: int,
    c_cache_pos: int,
  ) -> torch.Tensor:
    bz, seq_len = input_ids.shape
    word_embd = self.word_embeddings(input_ids) * self.embed_scale

    # position is cache_len: cache_len+seq_len
    cache_len = ks_cache.shape[2]
    pos_embd = self.position_embeddings(self.position_ids[:, cache_len:cache_len+seq_len])
    hidden_states = self.ln_begin(word_embd + pos_embd)
    '''
    new_ks_cache = []
    new_vs_cache = []
    new_kc_cache = []
    new_vc_cache = []
    '''
    for i, layer in enumerate(self.layers):
      output = layer(
        hidden_states,
        kv_states,
        self_attention_mask,
        cross_attention_mask,
        ks_cache[i, :, :s_cache_pos],
        vs_cache[i, :, :s_cache_pos],
        kc_cache[i, :, :c_cache_pos],
        vc_cache[i, :, :c_cache_pos],
      )
      hidden_states = output[0]
      '''
      new_ks_cache.append(output[1])
      new_vs_cache.append(output[2])
      new_kc_cache.append(output[3])
      new_vc_cache.append(output[4])
      '''
      ks_cache[i, :, :s_cache_pos+1] = output[1]
      vs_cache[i, :, :s_cache_pos+1] = output[2]
      kc_cache[i] = output[3]
      vc_cache[i] = output[4]

    out = self.ln_end(hidden_states)

    '''
    new_ks_cache = torch.stack(new_ks_cache)
    new_vs_cache = torch.stack(new_vs_cache)
    new_kc_cache = torch.stack(new_kc_cache)
    new_vc_cache = torch.stack(new_vc_cache)
    return out, new_ks_cache, new_vs_cache, new_kc_cache, new_vc_cache
    '''
    return out, ks_cache, vs_cache, kc_cache, vc_cache, s_cache_pos+1, kc_cache.shape[2]


class MBartLMHead(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super().__init__()
    self.proj = nn.Linear(embed_dim, vocab_size, bias=False)

  def forward(self, hidden_states):
    logits = self.proj(hidden_states)
    return logits

class NougatDecoder(nn.Module):
  def __init__(self, embed_dim: int, num_layers: int, num_heads: int, vocab_size: int, max_len: int, scale_embedding: bool = False):
    super().__init__()

    self.decoder = MBartDecoder(
      embed_dim,
      num_layers,
      num_heads,
      vocab_size,
      max_len,
      scale_embedding,
    )
    self.lm_head = MBartLMHead(vocab_size, embed_dim)

  def forward(
    self,
    input_ids: torch.Tensor,
    self_attention_mask: torch.Tensor,
    kv_states: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    ks_cache: torch.Tensor,  # (n_layers, bz, seq_len, embed_dim)
    vs_cache: torch.Tensor,  # (n_layers, bz, seq_len, embed_dim)
    kc_cache: torch.Tensor,  # (n_layers, bz, seq_len, encoder_embed_dim)
    vc_cache: torch.Tensor,  # (n_layers, bz, seq_len, encoder_embed_dim)
    s_cache_pos: int,
    c_cache_pos
  ) -> torch.Tensor:
    hidden_states, ks_cache, vs_cache, kc_cache, vc_cache, s_cache_pos, c_cache_pos = self.decoder(
      input_ids,
      self_attention_mask,
      kv_states,
      cross_attention_mask,
      ks_cache,
      vs_cache,
      kc_cache,
      vc_cache,
      s_cache_pos,
      c_cache_pos,
    )
    logits = self.lm_head(hidden_states)
    return logits, (ks_cache, vs_cache, kc_cache, vc_cache, s_cache_pos, c_cache_pos)
