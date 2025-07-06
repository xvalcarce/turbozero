# Utils

In this directory are scripts for testing and benchmarking trained AlphaZero agents. Most of them only apply to the QuantumCompilation environment and are "quick and dirty" scripts.

## `network_test.py`

To test different neural network classes. 
With this script the idea is only to get a policy and value returned by a randomly initialized neural net. 
Can also be used to compute the amount of free parameters a nn has.

## `singleplayer_test.py`

Load pre-trained neural network and test compilation. To compile a target unitary, e.g. in an ipython shell:
```python
%run singleplayer_test.py
compile(unitary='CXX', locs=[0,1,2], key=jax.random.PRNGKey(420), run=100, deterministic_run=True) 
```

## `benchmark.py`

Load pre-trained neural network (defined in `singleplayer_test.py` and test compilation for randomly sampled circuit of different depth. To benchmark compilation capabilities on 100 random circuits of depth 10 and compare it to MCTS:
```python
from benchmark import benchmark
benchmark(10,100,jax.random.PRNGKey(0),max_steps=10,mcts=True)
```

Compare the performance of an Alphazero agent w.r.t the simulation budget (number of iterations):
```python
from benchmark import *
for i in range(1,11):
    print(f"N iters: {i*100}")
    t = time.time()
    succ = 0
    d = 0
    for j in range(10):
        ssd = benchmark_niters(jax.random.key(j),10,i*100,max_steps=10,num_games=10)
        succ += len(ssd.nonzero()[0])
        d += ssd.nonzero()[1].sum()
    print(f"\tTime: {time.time()-t}")
    print(f"\tSucc: {succ}")
    if succ:
        print(f"\tDept: {d/succ}")
```

Compare the performance of the saved agent at every saved checkpoint steps:
```python
from benchmark import *
for i,v in enumerate(all_variables):
    s = 0
    print(f"Step: {ck.all_steps()[i]}")
    d = 0
    for j in range(10):
        ssd = benchmark_variables(jax.random.key(j),v,20,max_steps=20,num_games=10)
        s += len(ssd.nonzero()[0])
        d += ssd.nonzero()[1].sum()
    print(f"\tSuccess: {s}")
    if s > 0:
        print(f"\tAvg Depth: {d/s}")
```
