# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step5 vision-language model."""

from collections.abc import Iterable

import torch

from vllm.models.step5.model import _begin_step5_weight_load

from .step3p7 import Step3p7ForConditionalGeneration


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    """Step5 VLM using the Step3p7 multimodal frontend and Step5 text model."""

    _enable_weights_track_by_default = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Start the text update even if its first bucket contains only vision weights.
        state = _begin_step5_weight_load(self.language_model, self.language_model.model)
        try:
            return super().load_weights(weights)
        except Exception:
            if state is not None:
                state.failed = True
            raise

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()
