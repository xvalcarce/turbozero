
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience


def az_default_loss_fn(params: chex.ArrayTree, train_state: TrainState, experience: BaseExperience, 
                       l2_reg_lambda: float = 0.0001, 
                       dropout_key: jax.random.PRNGKey = jax.random.key(seed=0)) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """ Implements the default AlphaZero loss function.
    
    = Policy Loss + Value Loss + L2 Regularization
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value
    
    Args:
    - `params`: the parameters of the neural network
    - `train_state`: flax TrainState (holds optimizer and other state)
    - `experience`: experience sampled from replay buffer
        - stores the observation, target policy, target value
    - `l2_reg_lambda`: L2 regularization weight (default = 1e-4)
    - `dropout_key` : PRNGs for dropout

    Returns:
    - (loss, (aux_metrics, updates))
        - `loss`: total loss
        - `aux_metrics`: auxiliary metrics (policy_loss, value_loss)
        - `updates`: optimizer updates
    """

    # get batch_stats if using batch_norm
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    # get dropout key
    # fold_in with train_state.step which is increased for every call to apply_gradients() (train.py)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=train_state.step)

    # get predictions
    (pred_policy, pred_value), updates = train_state.apply_fn(
        variables, 
        x=experience.observation_nn,
        train=True,
        mutable=mutables,
        rngs={'dropout': dropout_train_key}
    )

    # set invalid actions in policy to -inf
    pred_policy = jnp.where(
        experience.policy_mask,
        pred_policy,
        jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(pred_policy, experience.policy_weights).mean()
    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    target_value = experience.reward[jnp.arange(experience.reward.shape[0]), current_player]
    # compute MSE value loss
    value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()

    # compute L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda x: (x ** 2).sum(),
            params
        )
    )

    # total loss
    loss = policy_loss + value_loss + l2_reg
    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss
    }
    return loss, (aux_metrics, updates)
