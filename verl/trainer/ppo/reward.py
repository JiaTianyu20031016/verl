# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import inspect
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional
import numpy as np

import ray
import torch
from omegaconf import DictConfig
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
from verl.workers.fsdp_workers import _expand_to_token_level


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    assert function_name is not None

    module = sys.modules.get("custom_module", None)
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)

    
def compute_GAN_like_reward(neg_batch: DataProto, pos_batch: DataProto, mode: str = "sum"):
    """Compute pairwise GAN-like rewards for paired positive/negative samples.

    Args:
        neg_batch: Batch of negative samples.
        pos_batch: Batch of positive samples.
        mode: Reward post-processing mode:
            - "sum" (default): wins - losses (current behavior).
            - "binary": 1 if reward > 0 else -1 (invalid samples stay 0).
            - "normalized": scale sum-mode rewards to [-1, 1] by max |reward| in batch (invalid stay 0).
    """

    if mode is not None and mode.lower() not in {"sum", "binary"}:
        raise NotImplementedError(f"Unsupported GAN-like reward mode: {mode}")
    mode = (mode or "sum").lower()
    
    with torch.no_grad():
        neg_scores = torch.sum(neg_batch.batch["token_level_scores"], dim=-1)
        pos_scores = torch.sum(pos_batch.batch["token_level_scores"], dim=-1)

        neg_valids = neg_batch.non_tensor_batch.get("rm_valids", None)
        pos_valids = pos_batch.non_tensor_batch.get("rm_valids", None)
        if neg_valids is None:
            neg_valids = np.ones(len(neg_scores), dtype=bool)
        if pos_valids is None:
            pos_valids = np.ones(len(pos_scores), dtype=bool)

        neg_uid = neg_batch.non_tensor_batch.get("uid", None)
        pos_uid = pos_batch.non_tensor_batch.get("uid", None)
        if neg_uid is None or pos_uid is None:
            raise ValueError("UIDs must be provided in both neg_batch and pos_batch for GAN-like reward computation.")

        neg_rewards = torch.zeros_like(neg_scores)
        pos_rewards = torch.zeros_like(pos_scores)

        # group indices by uid
        uid_to_neg = defaultdict(list)
        uid_to_pos = defaultdict(list)
        for idx, uid in enumerate(neg_uid):
            uid_to_neg[uid].append(idx)
        for idx, uid in enumerate(pos_uid):
            uid_to_pos[uid].append(idx)

        # compute pairwise wins-losses within each uid group
        for uid in set(list(uid_to_neg.keys()) + list(uid_to_pos.keys())):
            # invalid samples are skipped
            neg_indices = [i for i in uid_to_neg.get(uid, []) if neg_valids[i]]
            pos_indices = [i for i in uid_to_pos.get(uid, []) if pos_valids[i]]

            if mode == 'sum':
                for i in pos_indices:
                    for j in neg_indices:
                        pos_rewards[i] += pos_scores[i] - neg_scores[j]
                        neg_rewards[j] += pos_scores[i] - neg_scores[j]
            else:
                for i in pos_indices:
                    for j in neg_indices:
                        if pos_scores[i] > neg_scores[j]:
                            pos_rewards[i] += 1.0
                            neg_rewards[j] += 1.0
                        elif pos_scores[i] < neg_scores[j]:
                            pos_rewards[i] -= 1.0
                            neg_rewards[j] -= 1.0


        # expand back to token level
        neg_batch.batch["token_level_scores"] = _expand_to_token_level(neg_batch, neg_rewards)
        pos_batch.batch["token_level_scores"] = _expand_to_token_level(pos_batch, pos_rewards)
