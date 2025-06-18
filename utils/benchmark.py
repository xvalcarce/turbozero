import jax
import jax.numpy as jnp

import quantum_compilation.quantumcompilation as qc

from functools import partial
from core.types import StepMetadata

from singleplayer_test import SinglePlayerGameState, game_deterministic, alphazero_deterministic, game, alphazero_test

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

def benchmark(depth,runs,key=jax.random.PRNGKey(0),max_steps=max_steps,stochastic_runs=10):
    length = [] # storing depth of successful deterministic runs
    length_s = [] #storing depth of successful stochastic_runs
    bench = partial(one_benchmark_deterministic, depth=depth, max_steps=max_steps)
    benchs = partial(one_benchmark_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs)
    sub_runs = 10 # number of runs in parallel
    r = runs//sub_runs
    for i in range(r):
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=sub_runs) 
        # Deterministic runs
        ssd = jax.vmap(bench)(keys)
        idx = jnp.nonzero(ssd)
        length += idx[1].tolist()
        # Stochastic runs
        failed_keys = keys[jnp.nonzero(ssd.sum(axis=1).squeeze(axis=1)-1)]
        sss = jax.vmap(benchs)(failed_keys)
        nz = jnp.nonzero(sss)
        ids = jnp.unique(nz[0]) # successful runs
        idx_s = [jnp.min(nz[2][nz[0] == i]).tolist() for i in ids] # extracting minimum depth
        length_s += idx_s
    print(f"Compiled {len(length)+len(length_s)} ({len(length)}+{len(length_s)}) / {runs} unitary.")
    d_det = sum(length)/len(length) if length else 0
    d_sto = sum(length_s)/len(length_s) if length_s else 0
    print(f"Average compiled depth {(len(length)*d_det+len(length_s)*d_sto)/(len(length)+len(length_s))}.")
    print(f"\t deterministic: {d_det}.")
    print(f"\t Stochastic: {d_sto}.")
    return length, length_s

to_julia = [6,7,8,15,16,17,18,19,20,9,10,11,12,13,14,2,4,0,5,1,3]

if __name__ == "__main__":
    for i in range(5,30):
        print("Depth: {i}")
        rd,rs = benchmark(i,100,max_steps=i,stochastic_runs=10)
