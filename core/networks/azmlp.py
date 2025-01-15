from dataclasses import dataclass
import flax.linen as nn

@dataclass
class AZMLPConfig:
    """Configuration for a two-headed multilayer perceptron model:
    - `num_actions`: number of actions (output size of the policy head)
    - `width`: number of neurons in each dense layer
    - `depth_common`: number of dense layers in the common trunk
    - `depth_phead`: number of dense layers in the policy head
    - `depth_vhead`: number of dense layers in the value head
    - `use_batch_norm`: whether to use batch normalization
    - `batch_norm_momentum`: momentum for batch normalization
    - `dropout_rate`: dropout rate to apply after each dense layer
    """
    policy_head_out_size: int
    width: int
    depth_common: int
    depth_phead: int = 1
    depth_vhead: int = 1
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.6
    dropout_rate: float = 0.0


def make_dense_layer(out_dim, use_batch_norm, batch_norm_momentum, dropout_rate, train):
    """Helper function to create a dense layer, optionally with batch normalization and dropout."""
    layers = [nn.Dense(out_dim)]
    if use_batch_norm:
        layers.append(nn.BatchNorm(momentum=batch_norm_momentum, use_running_average=not train))
    layers.append(nn.relu)
    if dropout_rate > 0:
        layers.append(nn.Dropout(rate=dropout_rate, deterministic=not train))
    return nn.Sequential(layers)


class AZMLP(nn.Module):
    """ Two-headed multilayer perceptron.
    - `config`: network configuration
    """
    config: AZMLPConfig

    @nn.compact
    def __call__(self, x, train: bool):
        # Flatten input
        x = x.reshape((x.shape[0], -1))

        # Common trunk
        x = make_dense_layer(self.config.width, self.config.use_batch_norm, self.config.batch_norm_momentum, self.config.dropout_rate, train)(x)
        for _ in range(self.config.depth_common):
            x = make_dense_layer(self.config.width, self.config.use_batch_norm, self.config.batch_norm_momentum, self.config.dropout_rate, train)(x)

        # Policy head
        policy = x
        for _ in range(self.config.depth_phead):
            policy = make_dense_layer(self.config.width, self.config.use_batch_norm, self.config.batch_norm_momentum, self.config.dropout_rate, train)(policy)
        policy = nn.Dense(self.config.policy_head_out_size)(policy)
        policy = nn.softmax(policy)

        # Value head
        value = x
        for _ in range(self.config.depth_vhead):
            value = make_dense_layer(self.config.width, self.config.use_batch_norm, self.config.batch_norm_momentum, self.config.dropout_rate, train)(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value
