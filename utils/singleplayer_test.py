import os
import time
import pickle
import shutil
import configparser
import optax
import qujax
import chex
import jax
import jax.numpy as jnp

from functools import partial
from importlib import reload
from pathlib import Path
from chex import dataclass

import orbax.checkpoint as ocp

from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.networks.aztransformer import AZTransformer, AZTransformerConfig
from core.networks.azresnettransformer import AZResnetTransformer, AZResnetTransformerConfig
from core.networks.azmlp import AZMLP, AZMLPConfig
from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.weighted_mcts import MCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn, make_nn_eval_fn_no_params_callable
from core.training.train import Trainer, TrainLoopOutput
from core.training.loss_fns import az_default_loss_fn
from core.types import StepMetadata

# Load configuration
# load_dir = "./data/25-02-08_13h05/"
# load_dir = "./train/data/25-02-16_19h43/" # all-to-all 3 qubits
# load_dir = "./train/data/25-02-20_12h18/" # all-to-all 3 qubits
# load_dir = "./train/data/25-02-22_11h50/" # all-to-all 3 qubits Resnet
# load_dir = "./train/data/25-02-25_09h37/" # all-to-all 3 qubits Resnet Transformer
# load_dir = "./train/data/25-02-27_11h42/" # all-to-all 3 qubits Resnet Transformer
# load_dir = "./train/data/transformer/ancilla/25-04-01_12h37/" # ancilla transformer 2qubits+1
# load_dir = "./train/data/resnet/ancilla/25-04-09_11h43/" # ancilla resnet 3qubits+1
load_dir = "./train/data/resnet/ancilla/25-04-14_15h03/" # ancilla resnet 3qubits+1
load_dir = "./train/data/resnet/ancilla/25-06-23_15h30/" # ancilla resnet 3qubits+1, smaller model
# load_dir = "./train/data/resnet/ancilla/25-06-16_10h37/" # ancilla resnet 3qubits+1
# load_dir = "./foo/data/transformer/ancilla/25-04-23_16h54/" # ancilla resnet 3qubits+1

abs_load_dir = str(Path(os.getcwd()).parent.absolute())+load_dir[1:]
config = configparser.ConfigParser()
config.read(abs_load_dir+"config.ini")

# Use quantum_compilation game config file
import quantum_compilation as q
path_qc = q.__file__.split("/")[:-2]
path_qc.append("config.ini")
path_qc = "/".join(path_qc)
shutil.copyfile(abs_load_dir+"qc_config.ini",path_qc)
reload(q)
import quantum_compilation.quantumcompilation as qc

# Quantum compilation environment
env = qc.QuantumCompilation()
max_steps = qc.DEPTH
M_TARGET_DEPTH = int(config["environment"]["init_m_target_depth"])

# Target unitary
mat = qujax.get_params_to_unitarytensor_func(['CX'],[[0,1]],[[]],qc.N_QUBITS)
#mat = qujax.get_params_to_unitarytensor_func(['CCX'],[[0,1,2]],[[]],qc.N_QUBITS)
TARGET_V = mat().reshape(qc.DIM_OBS,qc.DIM_OBS).astype(jnp.complex64)

# define environment dynamics functions
def _init_fn(key,v=TARGET_V):
    state = qc._init_u(v,qc.MAX_TARGET_DEPTH)
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

# Agent
arch = config.get("neuralnetwork", "architecture") 
if arch == "Resnet":
    network = AZResnet
    networkconfig = AZResnetConfig
    kernel_size=int(config["neuralnetwork"]["kernel_size"])
    nn = network(networkconfig(
        policy_head_out_size=env.num_actions,
        num_blocks=int(config["neuralnetwork"]["num_blocks"]),
        num_channels=int(config["neuralnetwork"]["num_channels"]),
        num_policy_channels=int(config["neuralnetwork"]["num_policy_channels"]),
        num_value_channels=int(config["neuralnetwork"]["num_value_channels"]),
        kernel_size=kernel_size,
        kernel_size_value=config.getint("neuralnetwork","kernel_size_value", fallback=kernel_size),
        kernel_size_policy=config.getint("neuralnetwork","kernel_size", fallback=kernel_size),
        batch_norm_momentum=config.getfloat("neuralnetwork","batch_norm_momentum"),
    ))
elif arch == "ResnetTransformer":
    network = AZResnetTransformer
    networkconfig = AZResnetTransformerConfig
    nn = network(networkconfig(
        policy_head_out_size=env.num_actions,
        num_blocks=int(config["neuralnetwork"]["num_blocks"]),
        num_channels=int(config["neuralnetwork"]["num_channels"]),
        num_policy_channels=int(config["neuralnetwork"]["num_policy_channels"]),
        num_value_channels=int(config["neuralnetwork"]["num_value_channels"]),
        kernel_size=int(config["neuralnetwork"]["kernel_size"]),
        batch_norm_momentum=config.getfloat("neuralnetwork","batch_norm_momentum"),
        num_transformer_heads=config.getint("neuralnetwork","num_transformer_heads"),
        transformer_mlp_dim=config.getint("neuralnetwork","transformer_mlp_dim"),
        transformer_embed_dim=config.getint("neuralnetwork","transformer_embed_dim"),
    ))
elif arch == "Transformer":
    network = AZTransformer
    networkconfig = AZTransformerConfig
    nn = network(networkconfig(
        policy_head_out_size=env.num_actions,
        num_blocks=int(config["neuralnetwork"]["num_blocks"]),
        num_heads=config.getint("neuralnetwork","num_heads"),
        mlp_dim=config.getint("neuralnetwork","mlp_dim"),
        embed_dim=config.getint("neuralnetwork","embed_dim"),
    ))
elif arch == "MLP":
    network = AZMLP
    networkconfig = AZMLPConfig
    nn = network(networkconfig(
        policy_head_out_size=env.num_actions,
        width = config.getint("neuralnetwork","width"),
        depth_common = config.getint("neuralnetwork","depth_common"),
        depth_phead = config.getint("neuralnetwork","depth_phead"),
        depth_vhead = config.getint("neuralnetwork","depth_vhead"),
        use_batch_norm = config.getboolean("neuralnetwork","use_batch_norm", fallback=True),
        batch_norm_momentum = config.getfloat("neuralnetwork","batch_norm_momentum"),
        dropout_rate = config.getfloat("neuralnetwork","dropout_rate"),
    ))
else:
    raise TypeError("Network not supported")

replay_memory = EpisodeReplayBuffer(capacity=int(config["replay_memory"]["capacity"]))

def state_to_nn_input(state):
    # pgx does this for us with state.observation!
    return state.observation

## AlphaZero and MCTS players

# Define AlphaZero evaluator for self-play
alphazero = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(nn, state_to_nn_input),
    num_iterations=int(config["alphazero_evaluation"]["num_iterations"]),
    max_nodes=int(config["alphazero_evaluation"]["max_nodes"]),
    dirichlet_alpha=float(config["alphazero_selfplay"]["dirichlet_alpha"]),
    dirichlet_epsilon=float(config["alphazero_selfplay"]["dirichlet_epsilon"]),
    temperature=float(config["alphazero_selfplay"]["temperature"]),
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(c=float(config["alphazero_selfplay"]["puct_c"])),
    discount=float(config["alphazero_selfplay"]["discount"]),
)

# Define AlphaZero evaluator for evaluation games
alphazero_test = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(nn, state_to_nn_input),
    num_iterations=400,
    max_nodes=1000,
    temperature=0.6,
    dirichlet_alpha=config.getfloat("alphazero_evaluation", "dirichlet_alpha", fallback=1.0),
    dirichlet_epsilon=float(config["alphazero_evaluation"]["dirichlet_epsilon"]),
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(c=float(config["alphazero_evaluation"]["puct_c"])),
    discount=float(config["alphazero_evaluation"]["discount"]),
)

# Define AlphaZero evaluator for evaluation games
alphazero_deterministic = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(nn, state_to_nn_input),
    num_iterations=400,
    max_nodes=1000,
    temperature=0.0,
    dirichlet_alpha=config.getfloat("alphazero_evaluation", "dirichlet_alpha", fallback=1.0),
    dirichlet_epsilon=float(config["alphazero_evaluation"]["dirichlet_epsilon"]),
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(c=float(config["alphazero_evaluation"]["puct_c"])),
    discount=float(config["alphazero_evaluation"]["discount"]),
)

def fid_eval(obs):
    obs = obs[0,0]+obs[1,1]*1j
    f = jnp.square(jnp.abs(obs.trace()))/qc.FID_RENORM > qc.FIDELTY
    return jnp.ones((1,env.num_actions)), jnp.array([f],dtype=jnp.float32)

fid_baseline_eval_fn = make_nn_eval_fn_no_params_callable(fid_eval, state_to_nn_input)

mcts_baseline = AlphaZero(MCTS)(
        eval_fn=fid_baseline_eval_fn,
        num_iterations = 400,
        max_nodes = 1000,
        branching_factor = env.num_actions,
        action_selector = PUCTSelector(c=float(config["alphazero_evaluation"]["puct_c"])),
        temperature = 0.0,
        discount = 1.0
)

mcts_baseline_stochastic = AlphaZero(MCTS)(
        eval_fn=fid_baseline_eval_fn,
        num_iterations = 400,
        max_nodes = 1000,
        branching_factor = env.num_actions,
        action_selector = PUCTSelector(c=float(config["alphazero_evaluation"]["puct_c"])),
        temperature = 0.6,
        discount = 1.0
)

# Initialize trainer
batch_size = int(config["trainer"]["batch_size"])
train_batch_size = int(config["trainer"]["train_batch_size"])
warmup_steps = int(config["trainer"]["warmup_steps"])
collection_steps_per_epoch = int(config["trainer"]["collection_steps_per_epoch"])
train_steps_per_epoch = batch_size * collection_steps_per_epoch // train_batch_size

opt = config.get("trainer", "optimizer") 
if opt =="sgd":
    optimizer = optax.sgd
elif opt == "adam":
    optimizer = optax.adam
elif opt == "adamw":
    optimizer = optax.adamw
else:
    raise TypeError("Not a valid optimizer (sgd, adam, adamw)")


# Dummy trainer
trainer = Trainer(
    batch_size=8,
    train_batch_size=8,
    warmup_steps=warmup_steps,
    collection_steps_per_epoch=8,
    train_steps_per_epoch=8,
    nn=nn,
    loss_fn=partial(az_default_loss_fn, l2_reg_lambda=float(config["trainer"]["l2_reg_lambda"])),
    optimizer=optimizer(float(config["trainer"]["optimizer_lr"])),
    evaluator=alphazero,
    memory_buffer=replay_memory,
    max_episode_steps=max_steps,
    env_step_fn=step_fn,
    env_init_fn=_init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers=[],
    evaluator_test=alphazero_test,
)

# Load agent from saved data
def loading() -> TrainLoopOutput:
    with open(abs_load_dir+'collection.pickle', 'rb') as f:
        collection_state = pickle.load(f)
    with open(abs_load_dir+'test_states.pickle', 'rb') as f:
        # Serialize and save the object to the file
        test_states = pickle.load(f)
    with open(abs_load_dir+'cur_epoch.pickle', 'rb') as f:
        # Serialize and save the object to the file
        cur_epoch = pickle.load(f)

    # Restore backup train state
    # Copy backed up checkpoint to ckpt_dir
    shutil.copytree(abs_load_dir+str(cur_epoch-1),trainer.ckpt_dir+"/"+str(cur_epoch-1),dirs_exist_ok=True)
    # Load train_state
    train_state = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, cur_epoch-1)
    
    # Build a TrainLoopOutput
    init_state = TrainLoopOutput(
        collection_state=collection_state,
        train_state=train_state,
        test_states=test_states,
        cur_epoch=cur_epoch)
    return init_state

def reshape_nested_dict(d):
    if isinstance(d, dict):
        return {k: reshape_nested_dict(v) for k, v in d.items()}
    elif isinstance(d, jnp.ndarray) and d.shape[0] == 1:  
        return d.squeeze(axis=0)  # Remove the first dimension
    else:
        return d  # Return unchanged if not a JAX array or has different shape

def print_circuit(gates, l):
    out = " ; ".join([qc.GATE_NAMES[g] for g in gates.tolist()[:l]])
    print(out)

# abstract tree for params restoring
key = jax.random.PRNGKey(0)
init_key, key = jax.random.split(key)
init_keys = jnp.tile(init_key[None], (trainer.num_devices, 1))
dummy_state = trainer.init_train_state(init_keys)

# restore nn params from latest training step
ck = ocp.CheckpointManager(abs_load_dir)
try:
    s = ck.restore(ck.latest_step(), args=ocp.args.StandardRestore(dummy_state, strict=False))
except:
    s = ck.restore(ck.latest_step(), items=dummy_state, restore_kwargs={'strict': False})
s = ck.restore(ck.latest_step(), args=ocp.args.StandardRestore(dummy_state, strict=False))
variables = {'params': s.params, 'batch_stats': s.batch_stats}
variables = reshape_nested_dict(variables) # squeeze num_devices

def restore_variables(i):
    try:
        s = ck.restore(i, args=ocp.args.StandardRestore(dummy_state, strict=False))
    except:
        s = ck.restore(i, items=dummy_state, restore_kwargs={'strict': False})
    s = ck.restore(i, args=ocp.args.StandardRestore(dummy_state, strict=False))
    variables = {'params': s.params, 'batch_stats': s.batch_stats}
    variables = reshape_nested_dict(variables) # squeeze num_devices
    return variables

all_variables = [restore_variables(i) for i in ck.all_steps()]

# AZ agent
evaluator = alphazero
env_state, metadata = _init_fn(key)
eval_state = evaluator.init(template_embedding=env_state)

# Better single player class
@dataclass(frozen=True)
class SinglePlayerGameState:
    """Stores the state of a single-player game for two evaluators playing independently.
    - `key`: rng
    - `env_state`: The initial environment state.
    - `env_state_metadata`: Metadata associated with the initial environment state.
    - `eval_state_1`: The internal state of the first evaluator.
    - `eval_state_2`: The internal state of the second evaluator.
    - `completed_1`: Whether the first evaluator's game is completed.
    - `completed_2`: Whether the second evaluator's game is completed.
    - `outcome_1`: The final reward of the first evaluator.
    - `outcome_2`: The final reward of the second evaluator.
    """
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    env_state_metadata: StepMetadata
    eval_state: chex.ArrayTree
    completed: bool
    outcome: float

# A game
# @partial(jax.pmap, axis_name='p', static_broadcasted_argnums=(0,))
def game_step(state: SinglePlayerGameState, _, params: chex.ArrayTree, env_step_fn=step_fn, evaluator=alphazero):
    step_key, key = jax.random.split(state.key)
    # Evaluate and take action
    output = evaluator.evaluate(
        key=step_key,
        eval_state=state.eval_state,
        env_state=state.env_state,
        root_metadata=state.env_state_metadata,
        params=params,
        env_step_fn=env_step_fn
    )
    next_env_state, next_env_metadata = env_step_fn(state.env_state, output.action)
    terminated = next_env_metadata.terminated
    truncated = next_env_metadata.step > max_steps
    completed = terminated | truncated
    rewards = next_env_metadata.rewards
    eval_state = jax.lax.cond(
        completed,
        lambda _: state.eval_state,
        lambda _: evaluator.step(state.eval_state, output.action),
        None
    )
    state = state.replace(
            key=key,
            env_state = next_env_state,
            env_state_metadata = next_env_metadata,
            eval_state = eval_state,
            completed = completed,
            outcome = rewards)
    return state, state

game_step_ = partial(game_step, params=variables, env_step_fn=step_fn, evaluator=alphazero_test)
game_step_deterministic = partial(game_step, params=variables, env_step_fn=step_fn, evaluator=alphazero_deterministic)
game_step_mcts = partial(game_step, params=None, env_step_fn=step_fn, evaluator=mcts_baseline)
game_step_mcts_stochastic = partial(game_step, params=None, env_step_fn=step_fn, evaluator=mcts_baseline_stochastic)

def game(key, state, max_steps=max_steps):
    state = state.replace(key=key)
    state, collection_state = jax.lax.scan(
            game_step_,
            init=state,
            xs=None,
            length=max_steps
            )
    return collection_state

def game_deterministic(key, state, max_steps=max_steps):
    state = state.replace(key=key)
    state, collection_state = jax.lax.scan(
            game_step_deterministic,
            init=state,
            xs=None,
            length=max_steps
            )
    return collection_state

def game_mcts(key, state, max_steps=max_steps):
    state = state.replace(key=key)
    state, collection_state = jax.lax.scan(
            game_step_mcts,
            init=state,
            xs=None,
            length=max_steps
            )
    return collection_state

def game_mcts_stochastic(key, state, max_steps=max_steps):
    state = state.replace(key=key)
    state, collection_state = jax.lax.scan(
            game_step_mcts_stochastic,
            init=state,
            xs=None,
            length=max_steps
            )
    return collection_state

def compile(unitary='CX',locs=[0,1],run=10,batch_run=10,key=jax.random.PRNGKey(0),deterministic_run=False, max_steps=max_steps, hotstart=[]):
    mat = qujax.get_params_to_unitarytensor_func([unitary],[locs],[[]],qc.N_QUBITS)
    target_v = mat().reshape(qc.DIM_OBS,qc.DIM_OBS).astype(jnp.complex64)
    env_state, metadata = _init_fn(key,v=target_v)
    max_steps = max_steps-len(hotstart)
    for g in hotstart:
        env_state, metadata = step_fn(env_state,g)
    print("Compiling the unitary:")
    print(env_state._target_unitary)

    if deterministic_run:
        #deterministic run (temp=0.)
        eval_state = alphazero_deterministic.init(template_embedding=env_state)
        init_state = SinglePlayerGameState(key=key, 
                                           env_state=env_state, 
                                           env_state_metadata=metadata, 
                                           eval_state=eval_state, 
                                           completed=jnp.array(False, dtype=jnp.bool_), 
                                           outcome=jnp.array([0.0], dtype=jnp.float32))
        t = time.time()
        sd = game_deterministic(key, init_state, max_steps)
        print(f"Runtime {round(time.time()-t,2)}")
        idx = jnp.where(sd.outcome == 1)
        if idx[0].size == 0:
            print("No circuit found deterministically.")
            return False
        else:
            len_c = idx[0][1]
            print(f"Circuit found deterministically with depth {len_c+1}:")
            print_circuit(sd.env_state._circuit[len_c],len_c+1)
            return True

    # stochastic runs (temp=1.)
    eval_state = alphazero_test.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                       env_state=env_state, 
                                       env_state_metadata=metadata, 
                                       eval_state=eval_state, 
                                       completed=jnp.array(False, dtype=jnp.bool_), 
                                       outcome=jnp.array([0.0], dtype=jnp.float32))
    gg = partial(game, state=init_state, max_steps=max_steps)
    r = run//batch_run
    t = time.time()
    for ii in range(r):
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=batch_run) # 10 is reasonnable for 8GB of VRAM
        s = jax.vmap(gg)(keys)
        # extract indicies, non zero values
        idx = jnp.where(s.outcome == 1)
        if idx[0].size != 0:
            print(f"Runtime {round(time.time()-t,2)}")
            # element with smallest len
            min_c = jnp.argmin(idx[1])
            id_c = idx[0][min_c] # idices of cicruit
            len_c = idx[1][min_c] # length of circuit
            print(f"Circuit found with depth {len_c+1}:")
            print_circuit(s.env_state._circuit[id_c][len_c],len_c+1)
            return True
    print(f"Runtime {round(time.time()-t,2)}")
    print("No circuit found.")
    return False
