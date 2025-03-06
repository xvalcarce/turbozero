from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp

@dataclass
class AZTransformerConfig:
    """Configuration for AlphaZero Transformer model."""
    policy_head_out_size: int
    num_blocks: int  # Number of Transformer encoder blocks
    embed_dim: int  # Dimension of embeddings
    num_heads: int  # Number of attention heads
    mlp_dim: int  # MLP dimension inside Transformer blocks
    dropout_rate: float = 0.1


class TransformerBlock(nn.Module):
    """Single Transformer Encoder Block."""
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool):
        # Multi-head Self Attention
        x = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dropout_rate=self.dropout_rate
        )(x, x, deterministic=not train)

        # Feedforward MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.relu(y)
        y = nn.Dense(self.embed_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)

        return x + y


class AZTransformer(nn.Module):
    """Transformer-based AlphaZero Network."""
    config: AZTransformerConfig

    @nn.compact
    def __call__(self, x, train: bool):
        batch_size, NN, _ = x.shape

        # Linear projection to embedding dimension
        x = nn.Dense(self.config.embed_dim)(x)

        # Positional embedding (learned)
        pos_emb = self.param('pos_embedding', nn.initializers.normal(0.02), (NN, self.config.embed_dim))
        x = x + pos_emb[None, :, :]  # Add positional encoding (broadcast over batch)

        # Transformer Encoder
        for _ in range(self.config.num_blocks):
            x = TransformerBlock(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_dim=self.config.mlp_dim,
                dropout_rate=self.config.dropout_rate
            )(x, train=train)

        # Mean pooling (might need attention pooling for lager model)
        x = x.mean(axis=1)

        # Policy head - action probabilities
        policy = nn.Dense(self.config.policy_head_out_size)(x)
        policy = nn.softmax(policy)

        # Value head - scalar value
        value = nn.Dense(1)(x)
        value = nn.tanh(value)

        return policy, value
