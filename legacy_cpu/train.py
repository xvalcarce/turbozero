
import numpy as np
import time
import torch

from legacy_cpu.env import _2048Env
from legacy_cpu.mcts import MCTS_Evaluator
from core.memory import GameReplayMemory


def load_from_checkpoint(filename, model_class, load_replay_memory=True):
    run_tag = filename.split('_')[0]
    checkpoint = torch.load(filename)
    hypers = checkpoint['hypers']
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.share_memory()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate, weight_decay=hypers.weight_decay, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    history = checkpoint['history']
    memory = None
    if load_replay_memory:
        memory = checkpoint.get('memory')
    elif memory is None:
        memory = GameReplayMemory(hypers.replay_memory_size)
    
    return model, optimizer, hypers, history, memory, run_tag
    

def save_checkpoint(model, optimizer, hypers, history, memory, run_tag='', save_replay_memory=True):
    epoch = history.cur_epoch
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hypers': hypers,
        'history': history,
        'memory': memory if save_replay_memory else None,
        'model_type': str(type(model))
    }, f'{run_tag}_ep{epoch}.pt')



MOVE_MAP = {0: 'right', 1: 'up', 2: 'left', 3: 'down'}
def test_network(model, hypers, tensor_conversion_fn, debug_print=False):
    env = _2048Env()
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn, cpuct=hypers.mcts_c_puct, training=False)
    env.reset()
    model.eval()
    with torch.no_grad():
        moves = 0
        while True:
            start_time = time.time()
            probs, value = model(tensor_conversion_fn([env.board]))
            if debug_print:
                print(env.board)
            terminated, _, reward, mcts_probs, move, _ = mcts.choose_progression(hypers.mcts_iters_eval)
            moves += 1
            if debug_print:
                print(f'Time elapsed: {time.time() - start_time}s')
                print(f'Move #: {moves}')
                print(f'Move: {MOVE_MAP[move]}')
                print(f'Network Probs: {torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()}')
                print(f'MCTS Probs: {mcts_probs}')
                print(f'Network value: {value.item()}')
                print(f'Q Value: {np.sum(mcts.puct_node.pior_w) / mcts.puct_node.n}')
            if terminated:
                if debug_print:
                    print(f'Terminated, final reward = {reward}')
                break
    return reward, moves, env.get_highest_square(), env.get_score()

def train(samples, model, optimizer, tensor_conversion_fn, c_prob=5):
    model.train()
    obs, mcts_probs, rewards = zip(*samples)
    obs = tensor_conversion_fn(obs)
    mcts_probs = torch.from_numpy(np.array(mcts_probs))
    rewards = torch.from_numpy(np.array(rewards)).unsqueeze(1).float().log()
    optimizer.zero_grad(set_to_none=True)

    exp_probs, exp_rewards = model(obs)
    value_loss = torch.nn.functional.mse_loss(exp_rewards, rewards)
    prob_loss = c_prob * torch.nn.functional.cross_entropy(exp_probs, mcts_probs)
    
    acc = torch.eq(torch.argmax(exp_probs, dim=1), torch.argmax(mcts_probs, dim=1)).float().mean()

    loss = value_loss + prob_loss
    loss.backward()
    optimizer.step()
    return value_loss.item(), prob_loss.item(), loss.item(), acc.item()

def collect_episode(model, hypers, tensor_conversion_fn, epsilon=None):
    model.eval()
    training_examples = []
    env = _2048Env()
    env.reset()
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn=tensor_conversion_fn, cpuct=hypers.mcts_c_puct, epsilon=epsilon, training=True)
    moves = 0
    deviations = []
    with torch.no_grad():
        while True:
            # get inputs, reward, mcts probs, run n_iterations of MCTS
            terminated, obs, _, mcts_probs, _ = mcts.choose_progression(hypers.mcts_iters_train)
            moves += 1
            training_examples.append([obs, mcts_probs])
            if terminated:
                break
        reward = moves
        rem_reward = moves
        for example in training_examples:
            example.append(rem_reward)
            rem_reward -= 1

    return training_examples, reward, moves, env.get_highest_square(), np.mean(deviations), env.get_score()
