from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp


@dataclass
class AZResnetTransformerConfig:
    """Configuration for an hybrid Resnet + Transformer AlphaZero model:
    - `policy_head_out_size`: output size of the policy head (number of actions)
    - `num_blocks`: number of residual blocks
    - `num_channels`: number of channels in each residual block
    - `num_transformer_heads`: number of attention heads in the transformer
    - `transformer_mlp_dim`: dimension of the MLP layer in the transformer
    - `num_transformer_layers`: number of transformer layers
    """
    policy_head_out_size: int
    num_blocks: int
    num_channels: int
    num_transformer_heads: int = 4
    transformer_mlp_dim: int = 256
    transformer_embed_dim: int = 128
    num_policy_channels: int = 1
    num_value_channels: int = 2
    batch_norm_momentum: float = 1.0
    kernel_size: int = 3


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
    
class TransformerEncoder(nn.Module):
    """Transformer encoder block."""
    embed_dim: int
    num_heads: int
    dropout_rate: float
    mlp_dim: int

    @nn.compact
    def __call__(self, x, train: bool):
        # Multi-head self-attention with dropout
        attn_output = nn.SelfAttention(num_heads=self.num_heads, 
                                       kernel_init=nn.initializers.xavier_uniform(),
                                       deterministic=not train)(x)
        attn_output = nn.Dropout(rate=self.dropout_rate)(attn_output, deterministic=not train)
        
        # Add & Normalize (LayerNorm after residual connection)
        x = nn.LayerNorm()(x + attn_output)

        # Feedforward Network (MLP block) with dropout
        mlp_output = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        mlp_output = nn.relu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate)(mlp_output, deterministic=not train)
        mlp_output = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate)(mlp_output, deterministic=not train)

        # Add & Normalize (LayerNorm after residual connection)
        x = nn.LayerNorm()(x + mlp_output)
        
        return x

class PositionalEncoding(nn.Module):
    """Add positional encoding to the input tensor."""
    seq_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        pos = jnp.arange(0, self.seq_len)[:, None]
        dim = jnp.arange(0, self.embed_dim)[None, :]
        # Encoding with sin/cos
        enc = pos / jnp.power(10000, (2 * (dim // 2)) / self.embed_dim)
        # sin for even dimensions and cos for odd
        enc = jnp.where(dim % 2 == 0, jnp.sin(enc), jnp.cos(enc))
        return x + enc


class AZResnetTransformer(nn.Module):
    """Implements the AlphaZero ResNet+Transformer model.
    - `config`: network configuration"""
    config: AZResnetTransformerConfig

    @nn.compact
    def __call__(self, x, train: bool):
        # initial conv layer
        x = nn.Conv(features=self.config.num_channels, kernel_size=(self.config.kernel_size,self.config.kernel_size), strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(momentum=self.config.batch_norm_momentum, use_running_average=not train)(x)
        x = nn.relu(x)

        # residual blocks
        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.num_channels, momentum=self.config.batch_norm_momentum, kernel_size=self.config.kernel_size)(x, train=train)

        # Flattening and PositionalEncoding
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        x = PositionalEncoding(seq_len=height * width, embed_dim=channels)(x)

        # Transformer encoder with dropout
        x = TransformerEncoder(
            num_heads=self.config.num_transformer_heads,
            mlp_dim=self.config.transformer_mlp_dim,
            embed_dim=channels,
            dropout_rate=0.1,
        )(x, train=train)

        # Reshape back for convolutional policy/value heads
        x = x.reshape(batch_size, height, width, channels)

        # Policy and value heads
        policy = nn.Conv(features=self.config.num_policy_channels, kernel_size=(1, 1))(x)
        policy = policy.reshape((batch_size, -1))
        policy = nn.Dense(features=self.config.policy_head_out_size)(policy)
        policy = nn.softmax(policy)

        value = nn.Conv(features=self.config.num_value_channels, kernel_size=(1, 1))(x)
        value = value.reshape((batch_size, -1))
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)

        return policy, value
