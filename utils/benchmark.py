import jax
import jax.numpy as jnp

import quantum_compilation.quantumcompilation as qc

from functools import partial
from core.types import StepMetadata

from singleplayer_test import SinglePlayerGameState, game_deterministic, alphazero_deterministic

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
                 _target_depth = jnp.array(depth),
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

def one_benchmark(key,depth):
    env_state, metadata = _init_fn(key,depth=depth)
    eval_state = alphazero_deterministic.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    sd = game_deterministic(key, init_state)
    return sd.outcome

def benchmark(depth,runs,key=jax.random.PRNGKey(0)):
    length = []
    bench = partial(one_benchmark, depth=depth)
    r = runs//10
    for i in range(r):
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=10) 
        ssd = jax.vmap(bench)(keys)
        idx = jnp.nonzero(ssd)
        length += idx[1].tolist()
    print(f"Compiled {len(length)} / {runs} unitary.")
    print(f"Average compiled depth {sum(length)/len(length)}.")
    return length



