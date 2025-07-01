import time
import jax
import jax.numpy as jnp

import quantum_compilation.quantumcompilation as qc

from functools import partial
from core.types import StepMetadata

from singleplayer_test import SinglePlayerGameState, game_deterministic, alphazero_deterministic, game, alphazero_test, game_mcts, mcts_baseline, game_mcts_stochastic, mcts_baseline_stochastic

env = qc.QuantumCompilation()
max_steps = qc.DEPTH

def _init_fn(key,depth=10):
    circuit = qc.rand_cir(depth, key)
    v = jnp.eye(qc.DIM, dtype=jnp.complex64)
    v = jax.lax.fori_loop(0, depth, lambda i,v: jnp.matmul(qc.GATES[circuit[i]],v), v) 
    # This performs identity if N_ANCILLA == 0, else slice |0> in, |0> out on ancillaes
    v = jax.lax.slice(v, (0,0), (qc.DIM,qc.DIM), (qc.TWO_ANCILLA,qc.TWO_ANCILLA))
    # renormalize
    v = v/jnp.linalg.norm(v, ord=2) 
    state = qc.State(_target_unitary = v.conjugate().transpose(),
                 _target_circuit = circuit,
                 _target_depth = jnp.array(max_steps),
                 legal_action_mask = qc._legal_action_mask(circuit,0)) # for ancilla case, not trivial 
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

key = jax.random.PRNGKey(0)
env_state, metadata = _init_fn(key)

def one_benchmark_deterministic(key,depth,max_steps=max_steps):
    env_state, metadata = _init_fn(key,depth=depth)
    eval_state = alphazero_deterministic.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    sd = game_deterministic(key, init_state, max_steps=max_steps)
    return sd.outcome

def one_benchmark_stochastic(key,depth,max_steps=max_steps,runs=10):
    env_state, metadata = _init_fn(key,depth=depth)
    keys = jax.random.split(key, num=runs)
    eval_state = alphazero_test.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    gg = partial(game, state=init_state, max_steps=max_steps)
    sd = jax.vmap(gg)(keys)
    return sd.outcome

def one_benchmark_mcts(key, depth, max_steps=max_steps):
    env_state, metadata = _init_fn(key,depth=depth)
    eval_state = mcts_baseline.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    sd = game_mcts_stochastic(key, init_state, max_steps=max_steps)
    return sd.outcome

def one_benchmark_mcts_stochastic(key, depth, max_steps=max_steps,runs=10):
    env_state, metadata = _init_fn(key,depth=depth)
    keys = jax.random.split(key, num=runs)
    eval_state = mcts_baseline_stochastic.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    gg = partial(game_mcts_stochastic, state=init_state, max_steps=max_steps)
    sd = jax.vmap(gg)(keys)
    return sd.outcome

def benchmark(depth,runs,key=jax.random.PRNGKey(0),max_steps=max_steps,stochastic_runs=10, mcts=False):
    length_d = [] # storing depth of successful deterministic runs
    length_s = [] # storing depth of successful stochastic_runs
    length_md = [] # storing depth of successful mcts deterministic
    length_ms = [] # storing depth of successful mcts t>0
    bench_d = partial(one_benchmark_deterministic, depth=depth, max_steps=max_steps) #benchmark az deterministic
    bench_s = partial(one_benchmark_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) # benchmark az stochastic
    bench_md = partial(one_benchmark_mcts, depth=depth, max_steps=max_steps) #benchmark mcts t=0
    bench_ms = partial(one_benchmark_mcts_stochastic, depth=depth, max_steps=max_steps) #benchmark mcts t>0
    sub_runs = 10 # number of runs in parallel
    r = runs//sub_runs
    t = time.time()
    for i in range(r):
        print(f"{i*sub_runs}/{runs}")
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=sub_runs) 
        # Deterministic runs
        ssd = jax.vmap(bench_d)(keys)
        idx = jnp.nonzero(ssd)
        length_d += idx[1].tolist()
        print(f"\tAZ deterministic: {len(idx[1].tolist())} ({time.time()-t})")
        # Stochastic runs
        failed_keys = keys[jnp.nonzero(ssd.sum(axis=1).squeeze(axis=1)-1)]
        sss = jax.vmap(bench_s)(failed_keys)
        nz = jnp.nonzero(sss)
        ids = jnp.unique(nz[0]) # successful runs
        idx_s = [jnp.min(nz[2][nz[0] == i]).tolist() for i in ids] # extracting minimum depth
        length_s += idx_s
        print(f"\tAZ stochastic: {len(ids)} ({time.time()-t})")
        if mcts:
            # mcts t=0 runs
            ssmd = jax.vmap(bench_md)(keys)
            idx = jnp.nonzero(ssmd)
            length_md += idx[1].tolist()
            print(f"\tMCTS deterministic: {len(idx[1].tolist())} ({time.time()-t})")
            # mcts t>0 runs
            failed_keys = keys[jnp.nonzero(ssmd.sum(axis=1).squeeze(axis=1)-1)]
            ssms = jax.vmap(bench_ms)(failed_keys)
            nz = jnp.nonzero(ssms)
            ids = jnp.unique(nz[0]) # successful runs
            idx_s = [jnp.min(nz[2][nz[0] == i]).tolist() for i in ids] # extracting minimum depth
            length_ms += idx_s
            print(f"\tMCTS stochastic: {len(ids)} ({time.time()-t})")
    print(f"Compiled {len(length_d)+len(length_s)} ({len(length_d)}+{len(length_s)}) / {runs} unitary.")
    if mcts:
        print(f"(MCTS) Compiled {len(length_md)+len(length_ms)} ({len(length_md)}+{len(length_ms)}) / {runs} unitary.")
    d_det = sum(length_d)/len(length_d) if length_d else 0
    d_sto = sum(length_s)/len(length_s) if length_s else 0
    d_mdet = sum(length_md)/len(length_md) if length_md else 0
    d_msto = sum(length_ms)/len(length_ms) if length_ms else 0
    print(f"Average compiled depth {(len(length_d)*d_det+len(length_s)*d_sto)/(len(length_d)+len(length_s))}.")
    print(f"\t Deterministic: {d_det}.")
    print(f"\t Stochastic: {d_sto}.")
    if mcts:
        print(f"(MCTS) Average compiled depth {(len(length_md)*d_mdet+len(length_ms)*d_msto)/(len(length_md)+len(length_ms))}.")
        print(f"\t Deterministic: {d_mdet}.")
        print(f"\t Stochastic: {d_msto}.")
        return length_d, length_s, length_md, length_ms
    return length_d, length_s

to_julia = [6,7,8,15,16,17,18,19,20,9,10,11,12,13,14,2,4,0,5,1,3]

if __name__ == "__main__":
    for i in range(5,30):
        print(f"Depth: {i}")
        rd,rs = benchmark(i,100,max_steps=i,stochastic_runs=10)
