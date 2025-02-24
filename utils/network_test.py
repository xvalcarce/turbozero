import jax
import quantum_compilation as q
import quantum_compilation.quantumcompilation as qc


from core.networks.azvit import AZVisionTransformer, AZVisionTransformerConfig
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.types import StepMetadata

# Quantum compilation environment
env = qc.QuantumCompilation()
max_steps = qc.DEPTH

# define environment dynamics functions
def step_fn(state, action):
    state = env.step(state, action)
    metadata = StepMetadata(
        rewards = state.rewards,
        terminated = state.terminated,
        action_mask = state.legal_action_mask,
        cur_player_id = state.current_player,
        step = state._step_count
    )
    return state, metadata

def _init_fn(key):
    state = env._init(key,m_target_depth=10)
    observation = env.observe(state)
    state = state.replace(observation=observation)
    metadata = StepMetadata(
        rewards = state.rewards,
        terminated = state.terminated,
        action_mask = state.legal_action_mask,
        cur_player_id = state.current_player,
        step=state._step_count
    )
    return state, metadata

def state_to_nn_input(state):
    # pgx does this for us with state.observation!
    return state.observation

network = AZVisionTransformer
networkconfig = AZVisionTransformerConfig
nn = network(networkconfig(
    policy_head_out_size=env.num_actions,
    resnet_num_blocks = 5,
    resnet_num_channels = 128,
    transformer_num_heads = 8,
    transformer_num_layers = 4,
    transformer_mlp_dim = 256,
    transformer_patches_size = 3,
    transformer_hidden_size = 128,
    batch_norm_momentum = 0.9,
    kernel_size = 3
))

network = AZResnet
networkconfig = AZResnetConfig
nn = network(networkconfig(
    policy_head_out_size=env.num_actions,
    num_blocks=5,
    num_channels=128,
    num_policy_channels=64,
    num_value_channels=8
))

key = jax.random.PRNGKey(0)
key, dropout_key = jax.random.split(key, num=2)
# get template env state
sample_env_state, _ = _init_fn(jax.random.PRNGKey(0))
# get sample nn input
sample_obs = state_to_nn_input(sample_env_state)
# initialize nn parameters
variables = nn.lazy_init(key, sample_obs[None, ...], train=False)

policy,value = nn.apply(variables, sample_obs[None,...], train=False)

params = variables["params"]
# Count total params
total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Total Parameters: {total_params}")
