import time
import copy
import jax
import jax.numpy as jnp

import quantum_compilation.quantumcompilation as qc

from functools import partial
from core.types import StepMetadata

from singleplayer_test import SinglePlayerGameState, game_deterministic, alphazero_deterministic, game, alphazero_test, game_mcts, mcts_baseline, game_mcts_stochastic, mcts_baseline_stochastic
from singleplayer_test import game_step, step_fn, variables
from singleplayer_test import all_variables, ck

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

def one_benchmark_mcts_stochastic_while(key, depth, max_steps=max_steps, runs=10):
    env_state, metadata = _init_fn(key,depth=depth)
    key, _ = jax.random.split(key)
    eval_state = mcts_baseline_stochastic.init(template_embedding=env_state)
    init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
    gg = partial(game_mcts_stochastic, state=init_state, max_steps=max_steps)
    state = gg(key)
    def fail(state):
        state, i = state
        fail_state = jax.lax.cond(state.completed.any(),
                            lambda _ : False,
                            lambda _ : True,
                            operand=None)
        max_runs = jax.lax.cond(i<runs,
                                lambda _ : True,
                                lambda _ : False,
                                operand=None)
        return fail_state*max_runs
    def body_fn(state):
        state, i = state
        key = jax.random.fold_in(jax.random.key(0), i)
        state = gg(key)
        return (state, i+1)
    final_state, i = jax.lax.while_loop(fail, body_fn, (state, 1)) 
    return final_state, i

def benchmark_niters(key,depth,n_iters,max_steps=max_steps,num_games=10):
    az = copy.copy(alphazero_deterministic)
    az.num_iterations = n_iters
    game_step_niters = partial(game_step, params=variables, env_step_fn=step_fn, evaluator=az)
    def game_niters(key, state, max_steps=max_steps):
        state = state.replace(key=key)
        state, collection_state = jax.lax.scan(
                game_step_niters,
                init=state,
                xs=None,
                length=max_steps
                )
        return collection_state
    def bench(key, depth, max_steps):
        env_state, metadata = _init_fn(key,depth=depth)
        eval_state = az.init(template_embedding=env_state)
        init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
        sd = game_niters(key, init_state, max_steps=max_steps)
        return sd.outcome
    b = partial(bench,depth=depth,max_steps=max_steps)
    keys = jax.random.split(key, num=num_games)
    ssd = jax.vmap(b)(keys)
    return ssd

def benchmark_variables(key,variables,depth,n_iters=400,max_steps=max_steps,num_games=10):
    az = copy.copy(alphazero_deterministic)
    az.num_iterations = n_iters
    game_step_variables = partial(game_step, params=variables, env_step_fn=step_fn, evaluator=az)
    def game_variables(key, state, max_steps=max_steps):
        state = state.replace(key=key)
        state, collection_state = jax.lax.scan(
                game_step_variables,
                init=state,
                xs=None,
                length=max_steps
                )
        return collection_state
    def bench(key, depth, max_steps):
        env_state, metadata = _init_fn(key,depth=depth)
        eval_state = az.init(template_embedding=env_state)
        init_state = SinglePlayerGameState(key=key, 
                                  env_state=env_state, 
                                  env_state_metadata=metadata, 
                                  eval_state=eval_state, 
                                  completed=jnp.array(False, dtype=jnp.bool_), 
                                  outcome=jnp.array([0.0], dtype=jnp.float32))
        sd = game_variables(key, init_state, max_steps=max_steps)
        return sd.outcome
    b = partial(bench,depth=depth,max_steps=max_steps)
    keys = jax.random.split(key, num=num_games)
    ssd = jax.vmap(b)(keys)
    return ssd

def benchmark_vmap(depth,runs,key=jax.random.PRNGKey(0),max_steps=max_steps,stochastic_runs=10, mcts=False):
    length_d = [] # storing depth of successful deterministic runs
    length_s = [] # storing depth of successful stochastic_runs
    length_md = [] # storing depth of successful mcts deterministic
    length_ms = [] # storing depth of successful mcts t>0
    bench_d = partial(one_benchmark_deterministic, depth=depth, max_steps=max_steps) #benchmark az deterministic
    bench_s = partial(one_benchmark_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) # benchmark az stochastic
    bench_md = partial(one_benchmark_mcts, depth=depth, max_steps=max_steps) #benchmark mcts t=0
    bench_ms = partial(one_benchmark_mcts_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) #benchmark mcts t>0
    sub_runs = 10 # number of runs in parallel
    r = runs//sub_runs
    t = time.time()
    for i in range(r):
        print(f"{i*sub_runs}/{runs}")
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=sub_runs) 
        # Deterministic runs
        ssd = jax.vmap(bench_d)(keys)
        idx = jnp.where(ssd == 1.0)
        length_d += idx[1].tolist()
        print(f"\tAZ deterministic: {len(idx[1].tolist())} ({time.time()-t})")
        # Stochastic runs
        failed_keys = keys[jnp.nonzero(ssd.sum(axis=1).squeeze(axis=1)-1)]
        sss = jax.vmap(bench_s)(failed_keys)
        nz = jnp.where(sss == 1.0)
        ids = jnp.unique(nz[0]) # successful runs
        idx_s = [jnp.min(nz[2][nz[0] == i]).tolist() for i in ids] # extracting minimum depth
        length_s += idx_s
        print(f"\tAZ stochastic: {len(ids)} ({time.time()-t})")
        if mcts:
            # mcts t=0 runs
            ssmd = jax.vmap(bench_md)(keys)
            idx = jnp.where(ssmd == 1.0)
            length_md += idx[1].tolist()
            print(f"\tMCTS deterministic: {len(idx[1].tolist())} ({time.time()-t})")
            # mcts t>0 runs
            failed_keys = keys[jnp.nonzero(ssmd.sum(axis=1).squeeze(axis=1)-1)]
            ssms = jax.vmap(bench_ms)(failed_keys)
            nz = jnp.where(ssms == 1.0)
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


def benchmark(depth,runs,key=jax.random.PRNGKey(0),max_steps=max_steps,stochastic_runs=10, mcts=False):
    length_d = [] # storing depth of successful deterministic runs
    length_s = [] # storing depth of successful stochastic_runs
    length_md = [] # storing depth of successful mcts deterministic
    length_ms = [] # storing depth of successful mcts t>0
    bench_d = partial(one_benchmark_deterministic, depth=depth, max_steps=max_steps) #benchmark az deterministic
    bench_s = partial(one_benchmark_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) # benchmark az stochastic
    bench_md = partial(one_benchmark_mcts, depth=depth, max_steps=max_steps) #benchmark mcts t=0
    bench_ms = partial(one_benchmark_mcts_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) #benchmark mcts t>0
    sub_runs = 10 # number of runs in parallel
    r = runs//sub_runs
    t = time.time()
    for i in range(r):
        print(f"{i*sub_runs}/{runs}")
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=sub_runs) 
        # Deterministic runs
        ssd = jax.vmap(bench_d)(keys)
        idx = jnp.where(ssd == 1.0)
        length_d += idx[1].tolist()
        print(f"\tAZ deterministic: {len(idx[1].tolist())} ({time.time()-t})")
        # Stochastic runs
        failed_keys = keys[jnp.nonzero(ssd.sum(axis=1).squeeze(axis=1)-1)]
        j = 0
        for k in failed_keys:
            ss = bench_s(k)
            nz = jnp.where(ss == 1.0)
            if len(nz[0])> 0:
                idx_s = jnp.min(nz[1]).tolist()
                length_s += [idx_s]
                j += 1
        print(f"\tAZ stochastic: {j}/{len(failed_keys)} ({time.time()-t})")
        if mcts:
            # mcts t=0 runs
            ssmd = jax.vmap(bench_md)(keys)
            idx = jnp.where(ssmd == 1.0)
            length_md += idx[1].tolist()
            print(f"\tMCTS deterministic: {len(idx[1].tolist())} ({time.time()-t})")
            # mcts t>0 runs
            failed_keys = keys[jnp.nonzero(ssmd.sum(axis=1).squeeze(axis=1)-1)]
            j = 0
            for k in failed_keys:
                sms = bench_ms(key)
                nz = jnp.where(sms == 1.0)
                if len(nz[0])>0:
                    idx_s = jnp.min(nz[1]).tolist()
                    length_ms += [idx_s]
                    j+=1
            print(f"\tMCTS stochastic: {j}/{len(failed_keys)} ({time.time()-t})")
    print(f"Compiled {len(length_d)+len(length_s)} ({len(length_d)}+{len(length_s)}) / {runs} unitary.")
    if mcts:
        print(f"(MCTS) Compiled {len(length_md)+len(length_ms)} ({len(length_md)}+{len(length_ms)}) / {runs} unitary.")
    d_det = sum(length_d)/len(length_d) if length_d else 0
    d_sto = sum(length_s)/len(length_s) if length_s else 0
    d_mdet = sum(length_md)/len(length_md) if length_md else 0
    d_msto = sum(length_ms)/len(length_ms) if length_ms else 0
    if len(length_d)+len(length_s) > 0:
        print(f"Average compiled depth {(len(length_d)*d_det+len(length_s)*d_sto)/(len(length_d)+len(length_s))}.")
        print(f"\t Deterministic: {d_det}.")
        print(f"\t Stochastic: {d_sto}.")
    if mcts:
        if len(length_md)+len(length_ms) > 0:
            print(f"(MCTS) Average compiled depth {(len(length_md)*d_mdet+len(length_ms)*d_msto)/(len(length_md)+len(length_ms))}.")
            print(f"\t Deterministic: {d_mdet}.")
            print(f"\t Stochastic: {d_msto}.")
        return length_d, length_s, length_md, length_ms
    return length_d, length_s


def benchmark_novmap(depth,runs,key=jax.random.PRNGKey(0),max_steps=max_steps,stochastic_runs=10, mcts=False):
    length_d = [] # storing depth of successful deterministic runs
    length_s = [] # storing depth of successful stochastic_runs
    length_md = [] # storing depth of successful mcts deterministic
    length_ms = [] # storing depth of successful mcts t>0
    bench_d = partial(one_benchmark_deterministic, depth=depth, max_steps=max_steps) #benchmark az deterministic
    bench_s = partial(one_benchmark_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) # benchmark az stochastic
    bench_md = partial(one_benchmark_mcts, depth=depth, max_steps=max_steps) #benchmark mcts t=0
    bench_ms = partial(one_benchmark_mcts_stochastic, depth=depth, max_steps=max_steps, runs=stochastic_runs) #benchmark mcts t>0
    sub_runs = 10 # number of runs in parallel
    t = time.time()
    for i in range(runs):
        print(f"{i}/{runs}")
        key, _ = jax.random.split(key)
        # Deterministic runs
        sd = bench_d(key)
        idx = jnp.where(sd == 1.0)
        if len(idx[1]) > 0:
            length_d += idx[0].tolist()
            print(f"\t✓ AZ deterministic ({time.time()-t})")
        else:
            # Stochastic runs
            ss = bench_s(key)
            nz = jnp.where(ss == 1.0)
            if len(nz[0])> 0:
                idx_s = jnp.min(nz[1]).tolist()
                length_s += [idx_s]
                print(f"\t✓ AZ stochastic ({time.time()-t})")
        if mcts:
            # mcts t=0 runs
            smd = bench_md(key)
            idx = jnp.where(smd == 1.0)
            if len(idx[1]) >0:
                length_md += idx[0].tolist()
                print(f"\t✓ MCTS deterministic ({time.time()-t})")
            else:
                # mcts t>0 runs
                sms = bench_ms(key)
                nz = jnp.where(sms == 1.0)
                if len(nz[0])>0:
                    idx_s = jnp.min(nz[1]).tolist()
                    length_ms += [idx_s]
                print(f"\t✓ MCTS stochastic ({time.time()-t})")
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
