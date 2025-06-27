# 2025 skunkworxdark (https://github.com/skunkworxdark)

import math
from typing import Literal

import torch
from torch.linalg import norm
from typing_extensions import get_args

from invokeai.app.invocations.fields import FluxConditioningField, FluxReduxConditioningField, TensorField
from invokeai.app.invocations.flux_redux import FluxReduxOutput
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    invocation,
)

DOWNSAMPLING_FUNCTIONS = Literal["nearest", "bilinear", "bicubic", "area", "nearest-exact"]


# This node is derived from https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
@invocation(
    "flux_redux_downsampling",
    title="FLUX Redux Downsampling",
    tags=["conditioning", "flux", "redux", "downsampling"],
    category="conditioning",
    version="1.1.1",
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
        ge=0.0,
        le=3.0,
        default=1.0,
        description="Redux weight (0.0-3.0)",
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
            redux_cond=FluxReduxConditioningField(
                conditioning=TensorField(tensor_name=tensor_name), mask=self.redux_conditioning.mask
            )
        )


@invocation(
    "flux_scale_conditioning",
    title="Scale FLUX Conditioning",
    tags=["conditioning", "math", "scale", "flux"],
    category="conditioning",
    version="1.0.0",
)
class FluxScaleConditioningInvocation(BaseInvocation):
    """Scales a FLUX conditioning field by a factor."""

    conditioning: FluxConditioningField = InputField(
        description="FLUX Conditioning to scale",
        input=Input.Connection,
    )
    scale: float = InputField(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Scaling factor",
    )
    negative: bool = InputField(
        default=False,
        description="Scale negative conditioning",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        # Assuming only one conditioning info for simplicity. Handle multiple if necessary.
        assert len(cond_data.conditionings) == 1, "Scaling currently supports only single conditioning info"
        original_info = cond_data.conditionings[0]
        assert isinstance(original_info, FLUXConditioningInfo)

        if self.negative:
            self.scale = -self.scale

        scaled_clip_embeds = original_info.clip_embeds * self.scale
        scaled_t5_embeds = original_info.t5_embeds * self.scale

        new_info = FLUXConditioningInfo(clip_embeds=scaled_clip_embeds, t5_embeds=scaled_t5_embeds)
        new_cond_data = ConditioningFieldData(conditionings=[new_info])
        new_cond_name = context.conditioning.save(new_cond_data)

        return FluxConditioningOutput(
            conditioning=FluxConditioningField(
                conditioning_name=new_cond_name,
                mask=self.conditioning.mask,  # Pass through the original mask
            )
        )


@invocation(
    "flux_scale_redux_conditioning",
    title="Scale FLUX Redux Conditioning",
    tags=["conditioning", "math", "scale", "flux", "redux"],
    category="conditioning",
    version="1.0.0",
)
class FluxScaleReduxConditioningInvocation(BaseInvocation):
    """Scales a FLUX Redux conditioning field by a factor."""

    redux_conditioning: FluxReduxConditioningField = InputField(
        description="FLUX Redux Conditioning to scale",
        input=Input.Connection,
    )
    scale: float = InputField(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Scaling factor",
    )
    negative: bool = InputField(
        default=False,
        description="Scale negative conditioning",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        cond_tensor = context.tensors.load(self.redux_conditioning.conditioning.tensor_name)
        if self.negative:
            self.scale = -self.scale

        scaled_tensor = cond_tensor * self.scale
        new_tensor_name = context.tensors.save(scaled_tensor)

        return FluxReduxOutput(
            redux_cond=FluxReduxConditioningField(
                conditioning=TensorField(tensor_name=new_tensor_name), mask=self.redux_conditioning.mask
            )
        )


CONDITIONING_MATH_OPERATIONS = Literal["ADD", "SUB", "MUL", "DIV", "APPEND", "SPV", "NSPV", "LERP", "SLERP"]
CONDITIONING_MATH_LABELS = {
    "ADD": "Add (A + (Scale * B))",
    "SUB": "Subtract (A - (Scale * B))",
    "MUL": "Multiply (A * (Scale * B))",
    "DIV": "Divide (A / (Scale * B))",
    "APPEND": "Append Cat[A, B]",
    "SPV": "Scaled Projection Vector (A + (Scale * (Proj(A on B)))",
    "NSPV": "Negative Scaled Projection Vector (A - (Scale * Proj(A on B)))",
    "LERP": "Linear Interpolation A->B Scale= ",
    "SLERP": "Spherical Linear Interpolation A->B",
}


def _apply_conditioning_math(
    cond_a: torch.Tensor,
    cond_b: torch.Tensor,
    scale: float,
    operation: CONDITIONING_MATH_OPERATIONS,
    high_prec_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    assert cond_a.shape == cond_b.shape, (
        f"Tensor shapes must match for math operations a={cond_a.shape} b={cond_b.shape}"
    )

    # Store original properties
    original_dtype = cond_a.dtype
    a = cond_a.to(dtype=high_prec_dtype)
    b = cond_b.to(dtype=high_prec_dtype)

    epsilon = torch.finfo(high_prec_dtype).eps
    out = torch.zeros_like(a)

    match operation:
        case "ADD":
            out = a + (scale * b)
        case "SUB":
            out = a - (scale * b)
        case "MUL":
            out = a * (scale * b)
        case "DIV":
            out = a / (scale * b + epsilon)
        case "APPEND":
            out = torch.cat((a, b), dim=-1)
        case "SPV":
            dot_product = torch.mul(a, b).sum()
            norm_b_sq = torch.norm(b) ** 2
            proj_a_on_b = (dot_product / (norm_b_sq + epsilon)) * b
            out = a + scale * proj_a_on_b
        case "NSPV":
            dot_product = torch.mul(a, b).sum()
            norm_b_sq = torch.norm(b) ** 2
            proj_a_on_b = (dot_product / (norm_b_sq + epsilon)) * b
            out = a - scale * proj_a_on_b
        case "LERP":
            assert scale >= 0.0 and scale <= 1.0, "Scale must be between 0.0 and 1.0 for LERP"
            out = torch.lerp(a, b, scale, out=out)
        case "SLERP":
            assert scale >= 0.0 and scale <= 1.0, "Scale must be between 0.0 and 1.0 for SLERP"
            out = _slerp(a, b, scale)

    return out.to(dtype=original_dtype)


def _slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    weight: float,
    *,
    no_NaN: bool = True,
    DOT_THRESHOLD: float = 0.9995,
) -> torch.Tensor:
    """
    Spherical linear interpolation
    Args:
        v0: The starting vector (torch.Tensor).
        v1: The target vector (torch.Tensor).
        weight: The interpolation factor (float), where 0 represents `v0` and 1 represents `v1`.
        no_NaN: If True, replace potential NaNs in the output with zeros (default True).
        DOT_THRESHOLD: Threshold for considering vectors collinear and using LERP (default 0.9995).

    Returns:
        Interpolated vector between `v0` and `v1` with the same dtype as v0
    """

    # based on keturn's version from https://github.com/dunkeroni/InvokeAI_ConditioningMathNodes/pull/3/files

    tiny = torch.finfo(v0.dtype).tiny
    eps = torch.finfo(v0.dtype).eps

    # Normalize the vectors
    v0_norm = norm(v0, dim=-1).clamp_min_(tiny)
    v1_norm = norm(v1, dim=-1).clamp_min_(tiny)

    v0_normed = v0 / v0_norm.unsqueeze(-1)
    v1_normed = v1 / v1_norm.unsqueeze(-1)

    # Dot product
    dot = (v0_normed * v1_normed).sum(-1)
    dot = dot.clamp(-1.0, 1.0)  # Clamp for arccos stability
    dot_mag = dot.abs()

    # Identify cases for LERP vs SLERP
    gotta_lerp = dot_mag > DOT_THRESHOLD
    can_slerp = ~gotta_lerp

    # Initialize output tensor
    out = torch.zeros_like(v0)

    # --- Linear Interpolation (LERP) path ---
    if gotta_lerp.any():
        lerped = torch.lerp(v0, v1, weight)
        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # --- Spherical Linear Interpolation (SLERP) path ---
    if can_slerp.any():
        # Calculate angles
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        theta_t = theta_0 * weight

        # Check for division by zero in sin_theta_0
        sin_theta_0_safe = torch.where(
            sin_theta_0.abs() < eps,
            torch.full_like(sin_theta_0, eps),
            sin_theta_0,
        )

        # Calculate slerp coefficients
        s0 = (theta_0 - theta_t).sin() / sin_theta_0_safe
        s1 = theta_t.sin() / sin_theta_0_safe
        slerped = s0 * v0 + s1 * v1

        out = slerped.where(can_slerp.unsqueeze(-1), out)

    # Handle potential NaNs if requested (apply before casting back)
    if no_NaN:
        out = torch.nan_to_num(out)

    return out


@invocation(
    "flux_conditioning_math",
    title="FLUX Conditioning Math",
    tags=["conditioning", "math", "add", "subtract", "multiply", "divide", "flux"],
    category="conditioning",
    version="1.0.0",
)
class FluxConditioningMathOperationInvocation(BaseInvocation):
    """Performs a Math operation on two FLUX conditionings."""

    cond_a: FluxConditioningField = InputField(
        description="First FLUX Conditioning (A)",
        input=Input.Connection,
    )
    cond_b: FluxConditioningField = InputField(
        description="Second FLUX Conditioning (B)",
        input=Input.Connection,
    )
    operation: CONDITIONING_MATH_OPERATIONS = InputField(
        default=get_args(CONDITIONING_MATH_OPERATIONS)[0],  # "ADD",
        description="Operation to perform (A op B)",
        ui_choice_labels=CONDITIONING_MATH_LABELS,
    )
    scale: float = InputField(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Scaling factor",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        data_a = context.conditioning.load(self.cond_a.conditioning_name)
        data_b = context.conditioning.load(self.cond_b.conditioning_name)
        assert len(data_a.conditionings) == 1 and len(data_b.conditionings) == 1, (
            "Math operations support only single conditioning info per input"
        )
        info_a = data_a.conditionings[0]
        info_b = data_b.conditionings[0]
        assert isinstance(info_a, FLUXConditioningInfo) and isinstance(info_b, FLUXConditioningInfo)
        assert info_a.clip_embeds.shape == info_b.clip_embeds.shape, "CLIP embeds shapes must match"
        assert info_a.t5_embeds.shape == info_b.t5_embeds.shape, "T5 embeds shapes must match"

        result_clip_embeds = _apply_conditioning_math(
            info_a.clip_embeds, info_b.clip_embeds, self.scale, self.operation
        )
        result_t5_embeds = _apply_conditioning_math(info_a.t5_embeds, info_b.t5_embeds, self.scale, self.operation)

        new_info = FLUXConditioningInfo(clip_embeds=result_clip_embeds, t5_embeds=result_t5_embeds)
        new_cond_data = ConditioningFieldData(conditionings=[new_info])
        new_cond_name = context.conditioning.save(new_cond_data)

        output_field = FluxConditioningField(
            conditioning_name=new_cond_name,
            mask=self.cond_a.mask,  # Use mask from first input
        )
        return FluxConditioningOutput(conditioning=output_field)


@invocation(
    "flux_redux_conditioning_math",
    title="FLUX Redux Conditioning Math",
    tags=["conditioning", "math", "add", "subtract", "multiply", "divide", "flux", "redux"],
    category="conditioning",
    version="1.0.0",
)
class FluxReduxConditioningMathOperationInvocation(BaseInvocation):
    """Performs a Math operation on two FLUX Redux conditionings."""

    cond_a: FluxReduxConditioningField = InputField(
        description="First FLUX Redux Conditioning (A)",
        input=Input.Connection,
    )
    cond_b: FluxReduxConditioningField = InputField(
        description="Second FLUX Redux Conditioning (B)",
        input=Input.Connection,
    )
    operation: CONDITIONING_MATH_OPERATIONS = InputField(
        default=get_args(CONDITIONING_MATH_OPERATIONS)[0],  # 0 = ADD the first entry
        description="Operation to perform (A op B)",
        ui_choice_labels=CONDITIONING_MATH_LABELS,
    )
    scale: float = InputField(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Scaling factor",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        tensor_a = context.tensors.load(self.cond_a.conditioning.tensor_name)
        tensor_b = context.tensors.load(self.cond_b.conditioning.tensor_name)
        assert tensor_a.shape == tensor_b.shape, "Tensor shapes must match for math operations"

        result_tensor = _apply_conditioning_math(tensor_a, tensor_b, self.scale, self.operation)

        new_tensor_name = context.tensors.save(result_tensor)

        output_field = FluxReduxConditioningField(
            conditioning=TensorField(tensor_name=new_tensor_name),
            mask=self.cond_a.mask,  # Use mask from first input
        )
        return FluxReduxOutput(redux_cond=output_field)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a tensor to a unit vector (L2 norm of 1)."""
    # Calculate the norm along the last dimension, keeping the dimension for broadcasting
    tensor_norm = norm(tensor, dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero for zero-vectors
    epsilon = torch.finfo(tensor.dtype).eps
    # Divide by the norm
    return tensor / (tensor_norm + epsilon)


@invocation(
    "flux_normalize_conditioning",
    title="Normalize FLUX Conditioning",
    tags=["conditioning", "math", "normalize", "flux"],
    category="conditioning",
    version="1.0.0",
)
class FluxNormalizeConditioningInvocation(BaseInvocation):
    """Normalizes a FLUX conditioning field to a unit vector."""

    conditioning: FluxConditioningField = InputField(
        description="FLUX Conditioning to normalize",
        input=Input.Connection,
    )
    normalize_clip: bool = InputField(default=True, description="Normalize the CLIP embeddings")
    normalize_t5: bool = InputField(default=True, description="Normalize the T5 embeddings")

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1, "Normalization currently supports only single conditioning info"
        original_info = cond_data.conditionings[0]
        assert isinstance(original_info, FLUXConditioningInfo)

        clip_embeds = original_info.clip_embeds
        if self.normalize_clip:
            clip_embeds = _normalize_tensor(clip_embeds)

        t5_embeds = original_info.t5_embeds
        if self.normalize_t5:
            t5_embeds = _normalize_tensor(t5_embeds)

        new_info = FLUXConditioningInfo(clip_embeds=clip_embeds, t5_embeds=t5_embeds)
        new_cond_data = ConditioningFieldData(conditionings=[new_info])
        new_cond_name = context.conditioning.save(new_cond_data)

        return FluxConditioningOutput(
            conditioning=FluxConditioningField(
                conditioning_name=new_cond_name,
                mask=self.conditioning.mask,
            )
        )


@invocation(
    "flux_redux_normalize_conditioning",
    title="Normalize FLUX Redux Conditioning",
    tags=["conditioning", "math", "normalize", "flux", "redux"],
    category="conditioning",
    version="1.0.0",
)
class FluxReduxNormalizeConditioningInvocation(BaseInvocation):
    """Normalizes a FLUX Redux conditioning field to a unit vector."""

    redux_conditioning: FluxReduxConditioningField = InputField(
        description="FLUX Redux Conditioning to normalize",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        cond_tensor = context.tensors.load(self.redux_conditioning.conditioning.tensor_name)

        normalized_tensor = _normalize_tensor(cond_tensor)
        new_tensor_name = context.tensors.save(normalized_tensor)

        return FluxReduxOutput(
            redux_cond=FluxReduxConditioningField(
                conditioning=TensorField(tensor_name=new_tensor_name), mask=self.redux_conditioning.mask
            )
        )
