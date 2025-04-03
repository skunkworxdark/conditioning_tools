# 2025 skunkworxdark (https://github.com/skunkworxdark)

import math
from typing import Literal

import torch

from invokeai.app.invocations.fields import FluxReduxConditioningField, TensorField
from invokeai.app.invocations.flux_redux import FluxReduxOutput
from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    invocation,
)

DOWNSAMPLING_FUNCTIONS = Literal["nearest", "bilinear", "bicubic", "area", "nearest-exact"]


# This is derived from https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
@invocation(
    "flux_redux_downsampling",
    title="FLUX Redux Downsampling",
    tags=["image", "flux"],
    category="image",
    version="1.0.0",
)
class FluxReduxDownsamplingInvocation(BaseInvocation):
    """Downsampling Flux Redux conditioning"""

    redux_conditioning: FluxReduxConditioningField = InputField(
        description="FLUX Redux conditioning tensor.",
        input=Input.Connection,
    )
    downsampling_factor: int = InputField(
        ge=1,
        le=9,
        default=3,
        description="Redux Downsampling Factor (1-9)",
    )
    downsampling_function: DOWNSAMPLING_FUNCTIONS = InputField(
        default="area",
        description="Redux Downsampling Function",
    )
    weight: float = InputField(
        ge=0,
        le=1,
        default=1.0,
        description="Redux weight (0.0-1.0)",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        cond_redux = context.tensors.load(self.redux_conditioning.conditioning.tensor_name)

        rc = cond_redux.clone()
        (b, t, h) = rc.shape
        m = int(math.sqrt(t))
        if self.downsampling_factor > 1:
            rc = rc.view(b, m, m, h)

            rc = torch.nn.functional.interpolate(
                rc.transpose(1, -1),
                size=(m // self.downsampling_factor, m // self.downsampling_factor),
                mode=self.downsampling_function,
            )
            rc = rc.transpose(1, -1).reshape(b, -1, h)

        if self.weight != 1.0:
            rc = rc * self.weight * self.weight

        tensor_name = context.tensors.save(rc)
        return FluxReduxOutput(
            redux_cond=FluxReduxConditioningField(conditioning=TensorField(tensor_name=tensor_name), mask=self.redux_conditioning.mask)
        )
