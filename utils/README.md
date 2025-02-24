# Utils

In this directory are scripts for testing purpose. Most of them only apply to the QuantumCompilation environment and are "quick and dirty" scripts.

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

Load pre-trained neural network (defined in `singleplayer_test.py` and test compilation for randomly sampled circuit of different depth. To benchmark compilation capabilities on depth 10 for 100 circuits:
```python
%run benchmark.py
benchmark(10,100)
```
