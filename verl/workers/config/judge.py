# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .actor import ActorConfig
from .rollout import RolloutConfig


@dataclass
class JudgeParamsConfig:
    """
    Judge-specific parameters (LLM-as-a-Judge prompt & scoring settings).

    These are orthogonal to actor/rollout configs. If a field is None, the
    implementation should fall back to sensible defaults.
    """

    # Prompt & score parsing configuration
    prompt_template: Optional[str] = None
    score_pattern: str = r"Score:\s*(\d+(?:\.\d+)?)"
    score_min: float = 0.0
    score_max: float = 10.0
    normalize: bool = True

    # Micro-batch size for judge-side text generation (optional)
    micro_batch_size_per_gpu: Optional[int] = None


@dataclass
class JudgeConfig:
    """
    Composite judge configuration that contains a full `actor` config and a
    full `rollout` config, plus judge-specific parameters.

    Merging policy (implemented by trainer/loader, not here):
    - If `judge.actor` is None or empty in user config, copy from top-level `actor`.
    - If `judge.rollout` is None or empty in user config, copy from top-level `rollout`.
    - `judge.params` overrides defaults only for provided fields.
    """

    actor: ActorConfig = field(default_factory=ActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    params: JudgeParamsConfig = field(default_factory=JudgeParamsConfig)


__all__ = ["JudgeParamsConfig", "JudgeConfig"]
