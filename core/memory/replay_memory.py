from dataclasses import dataclass
from typing import Tuple
from flax import struct
import jax.numpy as jnp
import jax
from functools import partial

def expand_dims_to_match(array_to_expand, reference_array):
    target_shape = list(reference_array.shape)
    expand_shape = [-1 if i < len(array_to_expand.shape) else 1 for i in range(len(target_shape))]
    return jnp.broadcast_to(array_to_expand.reshape(expand_shape), target_shape)

@dataclass
class EndRewardReplayBufferConfig:
    buff_type: str
    capacity: int
    sample_size: int
    batch_size: int
    reward_size: int

@struct.dataclass
class EndRewardReplayBufferState:
    next_index: jnp.ndarray # next index to write experience to
    next_reward_index: jnp.ndarray # next index to write reward to
    buffer: struct.PyTreeNode # buffer of experiences
    reward_buffer: jnp.ndarray # buffer of rewards
    needs_reward: jnp.ndarray # does experience need reward
    populated: jnp.ndarray # is experience populated
    key: jax.random.PRNGKey

class EndRewardReplayBuffer:
    def __init__(self,
        config: EndRewardReplayBufferConfig,
    ):
        self.config = config

    def init(self, key: jax.random.PRNGKey, template_experience: struct.PyTreeNode) -> EndRewardReplayBufferState:
        return init(key, template_experience, self.config.batch_size, self.config.capacity, self.config.reward_size)

    def add_experience(self, state: EndRewardReplayBufferState, experience: struct.PyTreeNode) -> EndRewardReplayBufferState:
        return add_experience(state, experience, self.config.batch_size, self.config.capacity)    

    def assign_rewards(self, state: EndRewardReplayBufferState, rewards: jnp.ndarray, select_batch: jnp.ndarray) -> EndRewardReplayBufferState:
        return assign_rewards(state, rewards, select_batch.astype(jnp.bool_))

    def sample(self, state: EndRewardReplayBufferState) -> Tuple[EndRewardReplayBufferState, struct.PyTreeNode]:
        return sample(state, self.config.batch_size, self.config.capacity, self.config.sample_size)
    
    def truncate(self, state: EndRewardReplayBufferState, select_batch: jnp.ndarray) -> EndRewardReplayBufferState:
        return truncate(state, select_batch)



@partial(jax.jit, static_argnums=(2,3))
def add_experience(
    buffer_state: EndRewardReplayBufferState,
    experience: struct.PyTreeNode,
    batch_size: int,
    capacity: int,
) -> EndRewardReplayBufferState:
    
    def add_item(items, new_item):
        return items.at[jnp.arange(batch_size), buffer_state.next_index].set(new_item)

    return buffer_state.replace(
        buffer = jax.tree_map(add_item, buffer_state.buffer, experience),
        next_index = (buffer_state.next_index + 1) % capacity,
        needs_reward = buffer_state.needs_reward.at[:, buffer_state.next_index, 0].set(True),
        populated = buffer_state.populated.at[:, buffer_state.next_index, 0].set(True)
    )

@partial(jax.jit)
def assign_rewards(
    buffer_state: EndRewardReplayBufferState,
    rewards: jnp.ndarray,
    select_batch: jnp.ndarray,
) -> EndRewardReplayBufferState:
    return buffer_state.replace(
        reward_buffer = jnp.where(
            select_batch[..., None, None] & buffer_state.needs_reward,
            jnp.expand_dims(rewards, axis=1),
            buffer_state.reward_buffer
        ),
        next_reward_index = jnp.where(
            select_batch,
            buffer_state.next_index,
            buffer_state.next_reward_index
        ),
        needs_reward = jnp.where(
            select_batch[..., None, None],
            False,
            buffer_state.needs_reward
        )
    )

@partial(jax.jit, static_argnums=(1,2,3))
def sample(
    buffer_state: EndRewardReplayBufferState,
    batch_size: int,
    capacity: int,
    sample_size: int
) -> Tuple[EndRewardReplayBufferState, struct.PyTreeNode]:
    sample_key, new_key = jax.random.split(buffer_state.key)
    probs = ((~buffer_state.needs_reward).reshape(-1) * buffer_state.populated.reshape(-1)).astype(jnp.float32)
    indices = jax.random.choice(
        sample_key,
        capacity * batch_size,
        shape=(sample_size,),
        replace=False,
        p = probs / probs.sum()
    )
    batch_indices = indices // capacity
    item_indices = indices % capacity

    return buffer_state.replace(key=new_key), jax.tree_util.tree_map(
        lambda x: x[batch_indices, item_indices],
        buffer_state.buffer
    ), buffer_state.reward_buffer[batch_indices, item_indices]


@jax.jit
def truncate(
    buffer_state: EndRewardReplayBufferState,
    select_batch: jnp.ndarray,
) -> EndRewardReplayBufferState:
    
    def truncate_items(items):
        return jnp.where(
            select_batch.reshape(-1, *([1] * (items.ndim - 1))),
            items * (~buffer_state.needs_reward),
            items
        )

    return buffer_state.replace(
        buffer = jax.tree_map(truncate_items, buffer_state.buffer),
        needs_reward = truncate_items(buffer_state.needs_reward),
        populated = truncate_items(buffer_state.populated),
        next_index = jnp.where(
            select_batch,
            buffer_state.next_reward_index,
            buffer_state.next_index
        )
    )

@partial(jax.jit, static_argnums=(2,3,4))
def init(
    key: jax.random.PRNGKey,
    template_experience: struct.PyTreeNode,
    batch_size: int,
    capacity: int,
    reward_size: int = 1
) -> EndRewardReplayBufferState:
    experience = jax.tree_map(
        lambda x: jnp.broadcast_to(
            x, (batch_size, capacity, *x.shape)
        ),
        template_experience,
    )

    return EndRewardReplayBufferState(
        next_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        next_reward_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        reward_buffer=jnp.zeros((batch_size, capacity, reward_size), dtype=jnp.float32),
        buffer=experience,
        needs_reward=jnp.zeros((batch_size, capacity, 1), dtype=jnp.bool_),
        populated=jnp.zeros((batch_size, capacity, 1), dtype=jnp.bool_),
        key=key
    )