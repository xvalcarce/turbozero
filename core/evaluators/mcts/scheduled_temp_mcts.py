
from typing import Dict, Tuple
from functools import partial
from jax import Array

import chex
import jax
import jax.numpy as jnp

from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import MCTSTree, MCTSOutput
from core.types import EnvStepFn, StepMetadata


class ScheduledTemperatureMCTS(MCTS):
    """
       MCTS with a changing temperature as a function of steps.
    """

    def __init__(self, d_temperature: Array, *args, **kwargs):
        """Initializes a WeightedMCTS evaluator.
        
        Args:
        - `d_temperature`: temperature for each steps.
        """
        super().__init__(*args, **kwargs)
        self.d_temperature = d_temperature
        self.temperature = self.d_temperature[0]


    def get_config(self) -> Dict:
        """Returns the configuration of the ScheduledTemperatureMCTS evaluator. Used for logging."""
        return {
            "d_temperature": self.d_temperature,
            **super().get_config()
        }

    def set_temperature(self, root_metadata: StepMetadata):
        self.temperature = self.d_temperature[root_metadata.step]

    def sample_root_action(self, key: chex.PRNGKey, tree: MCTSTree) -> Tuple[int, chex.Array]:
        """Sample an action based on the root visit counts.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        
        Returns:
        - (Tuple[int, chex.Array]): sampled action, normalized policy weights
        """
        # get root visit counts
        action_visits = tree.get_child_data('n', tree.ROOT_INDEX)
        # normalize visit counts to get policy weights
        total_visits = action_visits.sum(axis=-1)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)
        policy_weights = jnp.where(total_visits > 0, policy_weights, 1 / self.branching_factor)

        temp = jnp.where(self.temperature == 0, self.tiebreak_noise, self.temperature)
        action, policy_weights = jax.lax.cond(self.temperature == 0,
                                              policy_noisy,
                                              policy,
                                              (key, policy_weights, temp))
        return action, policy_weights


    def evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete
        - `env_state`: current environment state
        - `root_metadata`: metadata for the root node of the tree
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSOutput): contains new tree state, selected action, root value, and policy weights
        """
        # store current state metadata in the root node
        key, root_key = jax.random.split(key)
        self.set_temperature(root_metadata)
        eval_state = self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata)
        # perform 'num_iterations' iterations of MCTS
        iterate = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        iterate_keys = jax.random.split(key, self.num_iterations)
        eval_state, _ = jax.lax.scan(lambda state, k: (iterate(k, state), None), eval_state, iterate_keys)
        # sample action based on root visit counts
        # (also get normalized policy weights for training purposes)
        action, policy_weights = self.sample_root_action(key, eval_state)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )

def policy_noisy(operand):
    key, policy_weights, tiebreak_noise = operand
    # break ties by adding small amount of noise
    noise = jax.random.uniform(key, shape=policy_weights.shape, maxval=tiebreak_noise)
    noisy_policy_weights = policy_weights + noise
    return jnp.argmax(noisy_policy_weights), policy_weights

def policy(operand):
    key, policy_weights, temperature = operand
    policy_weights_t = policy_weights ** (1/temperature)
    # re-normalize 
    policy_weights_t /= policy_weights_t.sum()
    # sample action
    action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
    # return original policy weights (we train on the policy before temperature is applied)
    return action, policy_weights
