import os
import datetime
import pickle
import shutil
import configparser
import optax
from functools import partial
import quantum_compilation as q
import quantum_compilation.quantumcompilation as qc


from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.evaluators.alphazero import AlphaZero
from core.evaluators.mcts.weighted_mcts import WeightedMCTS, MCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn, make_nn_eval_fn_no_params_callable
from core.testing.two_player_tester import TwoPlayerTester
from core.testing.single_player_tester import SinglePlayerTester
from core.training.train import Trainer, TrainLoopOutput
from core.training.loss_fns import az_default_loss_fn
from core.types import StepMetadata

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")

# Quantum compilation environment
env = qc.QuantumCompilation()
max_steps = qc.DEPTH
MAX_TARGET_DEPTH = int(config["environment"]["init_max_target_depth"])

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
    state = env._init(key,max_target_depth=MAX_TARGET_DEPTH)
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

def init_fn(key):
    state = env.init(key)
    metadata = StepMetadata(
        rewards = state.rewards,
        terminated = state.terminated,
        action_mask = state.legal_action_mask,
        cur_player_id = state.current_player,
        step=state._step_count
    )
    return state, metadata

resnet = AZResnet(AZResnetConfig(
    policy_head_out_size=env.num_actions,
    num_blocks=int(config["resnet"]["num_blocks"]),
    num_channels=int(config["resnet"]["num_channels"]),
    num_policy_channels=int(config["resnet"]["num_policy_channels"]),
    num_value_channels=int(config["resnet"]["num_value_channels"]),
    kernel_size=int(config["resnet"]["kernel_size"]),
    batch_norm_momentum=float(config["resnet"]["batch_norm_momentum"]),
))

replay_memory = EpisodeReplayBuffer(capacity=int(config["replay_memory"]["capacity"]))

def state_to_nn_input(state):
    # pgx does this for us with state.observation!
    return state.observation

# Define AlphaZero evaluator for self-play
alphazero = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations=int(config["alphazero_selfplay"]["num_iterations"]),
    max_nodes=int(config["alphazero_selfplay"]["max_nodes"]),
    dirichlet_alpha=float(config["alphazero_selfplay"]["dirichlet_alpha"]),
    dirichlet_epsilon=float(config["alphazero_selfplay"]["dirichlet_epsilon"]),
    temperature=float(config["alphazero_selfplay"]["temperature"]),
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(c=float(config["alphazero_selfplay"]["puct_c"])),
    discount=float(config["alphazero_selfplay"]["discount"]),
)

# Define AlphaZero evaluator for evaluation games
alphazero_test = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations=int(config["alphazero_evaluation"]["num_iterations"]),
    max_nodes=int(config["alphazero_evaluation"]["max_nodes"]),
    temperature=float(config["alphazero_evaluation"]["temperature"]),
    dirichlet_epsilon=float(config["alphazero_evaluation"]["dirichlet_epsilon"]),
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(c=float(config["alphazero_evaluation"]["puct_c"])),
    discount=float(config["alphazero_evaluation"]["discount"]),
)

# initialize trainer
# Initialize trainer
batch_size = int(config["trainer"]["batch_size"])
train_batch_size = int(config["trainer"]["train_batch_size"])
warmup_steps = int(config["trainer"]["warmup_steps"])
collection_steps_per_epoch = int(config["trainer"]["collection_steps_per_epoch"])
train_steps_per_epoch = batch_size * collection_steps_per_epoch // train_batch_size

trainer = Trainer(
    batch_size=batch_size,
    train_batch_size=train_batch_size,
    warmup_steps=warmup_steps,
    collection_steps_per_epoch=collection_steps_per_epoch,
    train_steps_per_epoch=train_steps_per_epoch,
    nn=resnet,
    loss_fn=partial(az_default_loss_fn, l2_reg_lambda=float(config["trainer"]["l2_reg_lambda"])),
    optimizer=optax.adam(float(config["trainer"]["optimizer_lr"])),
    evaluator=alphazero,
    memory_buffer=replay_memory,
    max_episode_steps=max_steps,
    env_step_fn=step_fn,
    env_init_fn=_init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers=[SinglePlayerTester(num_episodes=100)],
    evaluator_test=alphazero_test,
)

# Saving

# Training state automaticallly saved in train_loop
# Making a backup in another directory for re-use
bckp_dir = config["saving"]["bckp_dir"]
if config.getboolean("saving", "use_date"):
    d = datetime.datetime.today().strftime("%y-%m-%d_%Hh%M")
    bckp_dir += d+"/"
os.makedirs(bckp_dir, exist_ok = True)
print(f"Saving data to {bckp_dir}")

# Save quantum_compilation game config file
path_qc = q.__file__.split("/")[:-2]
path_qc.append("config.ini")
path_qc = "/".join(path_qc)
shutil.copyfile(path_qc,bckp_dir+"qc_config.ini")
# Save AlphaZero config file
shutil.copyfile("./config.ini",bckp_dir+"config.ini")

def saving(trainer, output):
    shutil.copytree(trainer.ckpt_dir+"/"+str(output.cur_epoch-1),bckp_dir+str(output.cur_epoch-1),dirs_exist_ok=True)

    # Saving other relevant objects to continue training
    with open(bckp_dir+'collection.pickle', 'wb') as file:
        pickle.dump(output.collection_state, file)
    with open(bckp_dir+'test_states.pickle', 'wb') as file:
        pickle.dump(output.test_states, file)
    with open(bckp_dir+'cur_epoch.pickle', 'wb') as file:
        pickle.dump(output.cur_epoch, file)

# Loading
# Load relevant objects
def loading() -> TrainLoopOutput:
    with open(bckp_dir+'collection.pickle', 'rb') as f:
        collection_state = pickle.load(f)
    with open(bckp_dir+'test_states.pickle', 'rb') as f:
        # Serialize and save the object to the file
        test_states = pickle.load(f)
    with open(bckp_dir+'cur_epoch.pickle', 'rb') as f:
        # Serialize and save the object to the file
        cur_epoch = pickle.load(f)

    # Restore backup train state
    # Copy backed up checkpoint to ckpt_dir
    shutil.copytree(bckp_dir+str(cur_epoch-1),trainer.ckpt_dir+"/"+str(cur_epoch-1),dirs_exist_ok=True)
    # Load train_state
    train_state = trainer.load_train_state_from_checkpoint(trainer.ckpt_dir, cur_epoch-1)
    
    # Build a TrainLoopOutput
    init_state = TrainLoopOutput(
        collection_state=collection_state,
        train_state=train_state,
        test_states=test_states,
        cur_epoch=cur_epoch)
    return init_state

# First Training
num_epochs = int(config["trainer"]["num_epochs"])
output = trainer.train_loop(seed=0, num_epochs=num_epochs)
saving(trainer, output)
init_state = loading()
k = 1

# Increasing MAX_TARGET_DEPTH, fine-tuning everytime
for i in range(
        int(config["environment"]["init_max_target_depth"])+1, 
        int(config["environment"]["final_max_target_depth"]),
        int(config["environment"]["target_depth_increment"])):
    k+=1
    MAX_TARGET_DEPTH = i
    trainer = Trainer(
        batch_size = batch_size, # number of parallel environments to collect self-play games from
        train_batch_size = train_batch_size, # training minibatch size
        warmup_steps = warmup_steps,
        collection_steps_per_epoch = collection_steps_per_epoch,
        train_steps_per_epoch = train_steps_per_epoch,
        nn = resnet,
        loss_fn = partial(az_default_loss_fn, l2_reg_lambda = 0.0001),
        optimizer = optax.adam(1e-3),
        evaluator = alphazero,
        memory_buffer = replay_memory,
        max_episode_steps=max_steps,
        env_step_fn = step_fn,
        env_init_fn = _init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[SinglePlayerTester(num_episodes=100)],
        evaluator_test = alphazero_test,
        # wandb_project_name='weighted_mcts_test' 
    )
    output_continued = trainer.train_loop(seed=0, num_epochs=num_epochs*k, initial_state=init_state);
    saving(trainer, output_continued)
    init_state = loading()