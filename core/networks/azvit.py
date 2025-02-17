from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Callable, Optional, Tuple, Type

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


@dataclass
class AZVisionTransformerConfig:
    """Configuration for a Vision Transformer model:
    - `policy_head_out_size`: output size of the policy head (number of actions)
    - `num_blocks`: number of residual blocks
    - `num_channels`: number of channels in each residual block
    - `num_transformer_heads`: number of attention heads in the transformer
    - `transformer_mlp_dim`: dimension of the MLP layer in the transformer
    - `num_transformer_layers`: number of transformer layers
    """
    policy_head_out_size: int
    resnet_num_blocks: int = 2
    resnet_num_channels: int = 64
    transformer_num_heads: int = 4
    transformer_num_layers: int = 1
    transformer_mlp_dim: int = 256
    transformer_patches_size: int = 3
    transformer_hidden_size: int = 128
    batch_norm_momentum: float = 1.0
    kernel_size: int = 3
    mlp_heads: bool = False

class AddPositionEmbs(nn.Module):
  """Adds learnable positional embeddings.
  - pe_init: position embeddings init.
  See https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
  """
  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """(bs, len(seq), emb_dim) -> (bs, timesteps, in_dim)
    """
    assert inputs.ndim == 3
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
    return inputs + pe

class ResidualBlock(nn.Module):
    """Residual block for AlphaZero Transformer model.
    """
    channels: int
    momentum: float
    kernel_size: int

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(features=self.channels, kernel_size=(self.kernel_size,self.kernel_size), strides=(1,1), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(momentum=self.momentum, use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.channels, kernel_size=(self.kernel_size,self.kernel_size), strides=(1,1), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(momentum=self.momentum, use_running_average=not train)(y)
        return nn.relu(x + y)

class MLPBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: int
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, x, *, deterministic):
    """Applies Transformer MlpBlock module.

    Add batchnorm?"""
    actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(features=actual_out_dim, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
    output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
    return output

class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MLPBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
    return x + y
    
class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded

class AZVisionTransformer(nn.Module):
    """Implements the AlphaZero ResNet+Transformer model.
    - `config`: network configuration"""
    config: AZVisionTransformerConfig

    @nn.compact
    def __call__(self, x, train: bool):
        # initial conv layer
        x = nn.Conv(features=self.config.resnet_num_channels, kernel_size=(self.config.kernel_size,self.config.kernel_size), strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(momentum=self.config.batch_norm_momentum, use_running_average=not train)(x)
        x = nn.relu(x)

        # residual blocks
        for _ in range(self.config.resnet_num_blocks):
            x = ResidualBlock(channels=self.config.resnet_num_channels, momentum=self.config.batch_norm_momentum, kernel_size=self.config.kernel_size)(x, train=train)

        #Flattening using a conv (patch embedding)
        x = nn.Conv(features=self.config.transformer_hidden_size,
                    kernel_size=self.config.transformer_patches_size,
                    strides=self.config.transformer_patches_size,
                    padding='VALID',
                    name='embedding')(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)

        # Transformer
        x = Encoder(num_layers=self.config.transformer_num_layers, 
                    mlp_dim=self.config.transformer_mlp_dim, 
                    num_heads=self.config.transformer_num_heads)(x, train=train)
        x = x[:,0]

        if self.config.mlp_heads:
            policy = nn.Dense(features=self.config.policy_head_out_size)(x)
            policy = nn.softmax(policy)

            value = nn.Dense(features=1)(x)
            value = nn.tanh(value)
        else:
            policy = x[...,0]
            value = x[...,1:]

        return policy, value
