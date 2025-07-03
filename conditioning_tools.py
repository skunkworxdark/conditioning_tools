# 2025 skunkworxdark (https://github.com/skunkworxdark)

import re
from math import sqrt
from typing import Literal, Optional, Union

import torch
from torch.linalg import norm
from transformers import T5TokenizerFast
from typing_extensions import get_args

from invokeai.app.invocations.fields import (
    FluxConditioningField,
    FluxReduxConditioningField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.flux_redux import FluxReduxOutput
from invokeai.app.invocations.flux_text_encoder import FluxTextEncoderInvocation
from invokeai.app.invocations.model import ModelIdentifierField, T5EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    CLIPField,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)

from .prompt_ast_parser import NestedPromptParser, flatten_ast

DOWNSAMPLING_FUNCTIONS = Literal["nearest", "bilinear", "bicubic", "area", "nearest-exact"]


def _rescale_to_target_max_norm(tensor: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Rescales a tensor so that the vector with the largest L2 norm has a norm of `target_norm`."""
    # Calculate the L2 norm for each vector along the last dimension
    norms = torch.linalg.norm(tensor, dim=-1)
    # Find the maximum norm
    max_norm = torch.max(norms)
    # Add a small epsilon to avoid division by zero
    epsilon = torch.finfo(tensor.dtype).eps
    # If max_norm is close to zero, no need to scale
    if max_norm < epsilon:
        return tensor
    return tensor * (target_norm / (max_norm + epsilon))


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

        redux_cond_tensor = cond_redux.clone()
        (b, t, h) = redux_cond_tensor.shape
        side_length = int(sqrt(t))
        if self.downsampling_factor > 1:
            redux_cond_tensor = redux_cond_tensor.view(b, side_length, side_length, h)

            redux_cond_tensor = torch.nn.functional.interpolate(
                redux_cond_tensor.transpose(1, -1),
                size=(side_length // self.downsampling_factor, side_length // self.downsampling_factor),
                mode=self.downsampling_function,
            )
            redux_cond_tensor = redux_cond_tensor.transpose(1, -1).reshape(b, -1, h)

        if self.weight != 1.0:
            redux_cond_tensor = redux_cond_tensor * self.weight * self.weight

        tensor_name = context.tensors.save(redux_cond_tensor)
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
    # Store original properties
    original_dtype = cond_a.dtype
    a = cond_a.to(dtype=high_prec_dtype)
    b = cond_b.to(dtype=high_prec_dtype)

    epsilon = torch.finfo(high_prec_dtype).eps
    out = torch.zeros_like(a)

    if operation != "APPEND":
        assert cond_a.shape == cond_b.shape, (
            f"Tensor shapes must match for this math operation: a={cond_a.shape}, b={cond_b.shape}"
        )

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
            assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[2], (
                "Batch size and embedding dim must match for APPEND"
            )
            out = torch.cat((a, b), dim=1)
        case "SPV":
            # Project a onto b, then add the scaled projection to a
            dot_product = torch.sum(a * b, dim=-1, keepdim=True)
            norm_b_sq = (torch.norm(b, dim=-1, keepdim=True) ** 2).clamp_min(epsilon)
            proj_a_on_b = (dot_product / (norm_b_sq + epsilon)) * b
            out = a + scale * proj_a_on_b
        case "NSPV":
            # Project a onto b, then subtract the scaled projection from a
            dot_product = torch.sum(a * b, dim=-1, keepdim=True)
            norm_b_sq = (torch.norm(b, dim=-1, keepdim=True) ** 2).clamp_min(epsilon)
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
    version="1.1.0",
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
    rescale_target_norm: float = InputField(
        default=0.0,
        ge=0.0,
        description="If > 0, rescales the output embeddings to the target max norm. Set to 0 to disable.",
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

        result_clip_embeds = _apply_conditioning_math(
            info_a.clip_embeds, info_b.clip_embeds, self.scale, self.operation
        )
        result_t5_embeds = _apply_conditioning_math(info_a.t5_embeds, info_b.t5_embeds, self.scale, self.operation)

        if self.rescale_target_norm > 0.0:
            # Log and rescale CLIP embeds
            original_clip_max_norm = torch.max(torch.linalg.norm(result_clip_embeds, dim=-1))
            context.logger.info(f"Original CLIP max norm after math: {original_clip_max_norm.item():.4f}")
            result_clip_embeds = _rescale_to_target_max_norm(result_clip_embeds, self.rescale_target_norm)
            new_clip_max_norm = torch.max(torch.linalg.norm(result_clip_embeds, dim=-1))
            context.logger.info(
                f"Rescaled CLIP max norm to: {new_clip_max_norm.item():.4f} (target: {self.rescale_target_norm:.4f})"
            )

            # Log and rescale T5 embeds
            original_t5_max_norm = torch.max(torch.linalg.norm(result_t5_embeds, dim=-1))
            context.logger.info(f"Original T5 max norm after math: {original_t5_max_norm.item():.4f}")
            result_t5_embeds = _rescale_to_target_max_norm(result_t5_embeds, self.rescale_target_norm)
            new_t5_max_norm = torch.max(torch.linalg.norm(result_t5_embeds, dim=-1))
            context.logger.info(
                f"Rescaled T5 max norm to: {new_t5_max_norm.item():.4f} (target: {self.rescale_target_norm:.4f})"
            )

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
    version="1.1.0",
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
    rescale_target_norm: float = InputField(
        default=0.0,
        ge=0.0,
        description="If > 0, rescales the output embeddings to the target max norm. Set to 0 to disable.",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        tensor_a = context.tensors.load(self.cond_a.conditioning.tensor_name)
        tensor_b = context.tensors.load(self.cond_b.conditioning.tensor_name)

        result_tensor = _apply_conditioning_math(tensor_a, tensor_b, self.scale, self.operation)
        if self.rescale_target_norm > 0.0:
            original_max_norm = torch.max(torch.linalg.norm(result_tensor, dim=-1))
            context.logger.info(f"Original Redux max norm after math: {original_max_norm.item():.4f}")

            result_tensor = _rescale_to_target_max_norm(result_tensor, self.rescale_target_norm)

            new_max_norm = torch.max(torch.linalg.norm(result_tensor, dim=-1))
            context.logger.info(
                f"Rescaled Redux max norm to: {new_max_norm.item():.4f} (target: {self.rescale_target_norm:.4f})"
            )

        new_tensor_name = context.tensors.save(result_tensor)

        output_field = FluxReduxConditioningField(
            conditioning=TensorField(tensor_name=new_tensor_name),
            mask=self.cond_a.mask,  # Use mask from first input
        )
        return FluxReduxOutput(redux_cond=output_field)


@invocation(
    "flux_rescale_conditioning",
    title="Rescale FLUX Conditioning",
    tags=["conditioning", "math", "rescale", "flux"],
    category="conditioning",
    version="1.0.0",
)
class FluxRescaleConditioningInvocation(BaseInvocation):
    """Rescales a FLUX conditioning field to a target max norm."""

    conditioning: FluxConditioningField = InputField(
        description="FLUX Conditioning to rescale",
        input=Input.Connection,
    )
    clip_rescale: bool = InputField(
        default=True,
        description="Whether to rescale the CLIP embeddings.",
    )
    clip_target_norm: float = InputField(
        default=30.0,
        gt=0.0,
        description="The target max norm for the CLIP embeddings.",
    )
    t5_rescale: bool = InputField(
        default=True,
        description="Whether to rescale the T5 embeddings.",
    )
    t5_target_norm: float = InputField(
        default=10.0,
        gt=0.0,
        description="The target max norm for the T5 embeddings.",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1, "Rescaling currently supports only single conditioning info"
        original_info = cond_data.conditionings[0]
        assert isinstance(original_info, FLUXConditioningInfo)

        if self.clip_rescale:
            # Log and rescale CLIP embeds
            original_clip_max_norm = torch.max(torch.linalg.norm(original_info.clip_embeds, dim=-1))
            context.logger.info(f"Original CLIP max norm: {original_clip_max_norm.item():.4f}")
            clip_embeds = _rescale_to_target_max_norm(original_info.clip_embeds, self.clip_target_norm)
            new_clip_max_norm = torch.max(torch.linalg.norm(clip_embeds, dim=-1))
            context.logger.info(
                f"Rescaled CLIP max norm to: {new_clip_max_norm.item():.4f} (target: {self.clip_target_norm:.4f})"
            )
        else:
            clip_embeds = original_info.clip_embeds

        if self.t5_rescale:
            # Log and rescale T5 embeds
            original_t5_max_norm = torch.max(torch.linalg.norm(original_info.t5_embeds, dim=-1))
            context.logger.info(f"Original T5 max norm: {original_t5_max_norm.item():.4f}")
            t5_embeds = _rescale_to_target_max_norm(original_info.t5_embeds, self.t5_target_norm)
            new_t5_max_norm = torch.max(torch.linalg.norm(t5_embeds, dim=-1))
            context.logger.info(
                f"Rescaled T5 max norm to: {new_t5_max_norm.item():.4f} (target: {self.t5_target_norm:.4f})"
            )
        else:
            t5_embeds = original_info.t5_embeds

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
    "flux_redux_rescale_conditioning",
    title="Rescale FLUX Redux Conditioning",
    tags=["conditioning", "math", "rescale", "flux", "redux"],
    category="conditioning",
    version="1.0.0",
)
class FluxReduxRescaleConditioningInvocation(BaseInvocation):
    """Rescales a FLUX Redux conditioning field to a target max norm."""

    redux_conditioning: FluxReduxConditioningField = InputField(
        description="FLUX Redux Conditioning to rescale",
        input=Input.Connection,
    )
    target_norm: float = InputField(
        default=1.0,
        gt=0.0,
        description="The target max norm for the conditioning.",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        cond_tensor = context.tensors.load(self.redux_conditioning.conditioning.tensor_name)

        original_max_norm = torch.max(torch.linalg.norm(cond_tensor, dim=-1))
        context.logger.info(f"Original Redux max norm: {original_max_norm.item():.4f}")

        rescaled_tensor = _rescale_to_target_max_norm(cond_tensor, self.target_norm)

        new_max_norm = torch.max(torch.linalg.norm(rescaled_tensor, dim=-1))
        context.logger.info(f"Rescaled Redux max norm to: {new_max_norm.item():.4f} (target: {self.target_norm:.4f})")

        new_tensor_name = context.tensors.save(rescaled_tensor)

        return FluxReduxOutput(
            redux_cond=FluxReduxConditioningField(
                conditioning=TensorField(tensor_name=new_tensor_name), mask=self.redux_conditioning.mask
            )
        )


@invocation(
    "flux_scale_prompt_section",
    title="Scale FLUX Prompt Section(s)",
    tags=["conditioning", "prompt", "scale", "flux"],
    category="conditioning",
    version="1.3.0",
)
class FluxScalePromptSectionInvocation(BaseInvocation):
    """Scales one or more sections of a FLUX prompt conditioning."""

    conditioning: FluxConditioningField = InputField(
        description="FLUX Conditioning to modify",
        input=Input.Connection,
    )
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description="T5 Encoder model and tokenizer used for the original conditioning.",
        input=Input.Connection,
    )
    prompt: str = InputField(
        description="The full prompt text used for the original conditioning.",
        ui_component=UIComponent.Textarea,
    )
    prompt_section: Union[str, list[str]] = InputField(
        description="The section or sections of the prompt to scale.",
    )
    scale: Union[float, list[float]] = InputField(
        default=1.0, description="The scaling factor or factors for the prompt section(s)."
    )
    positions: Optional[Union[int, list[int]]] = InputField(
        default=None,
        description="The start character position(s) of the section(s) to scale. If provided, this is used to locate the section(s) instead of searching.",
        input=Input.Connection,
    )
    rescale_output: bool = InputField(
        default=False,
        description="Rescales the output T5 embeddings to have the same max vector norm as the original conditioning.",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        original_info = cond_data.conditionings[0]
        assert isinstance(original_info, FLUXConditioningInfo)

        sections: list[str] = self.prompt_section if isinstance(self.prompt_section, list) else [self.prompt_section]
        scales: list[float] = self.scale if isinstance(self.scale, list) else [self.scale]

        positions_list: Optional[list[int]] = None
        if self.positions is not None:
            positions_list = [self.positions] if isinstance(self.positions, int) else self.positions
            if len(positions_list) != len(sections):
                raise ValueError("The number of prompt sections must match the number of positions.")

        num_items_to_scale = len(sections)

        if len(scales) == 1 and num_items_to_scale > 1:
            scales = scales * num_items_to_scale

        if len(scales) != num_items_to_scale:
            raise ValueError("The number of scales must match the number of sections to be scaled.")

        new_t5_embeds = self._process_embeds(
            context,
            original_info.t5_embeds,
            self.t5_encoder.tokenizer,
            self.prompt,
            sections,
            scales,
            positions_list,
        )

        if self.rescale_output:
            original_norms = torch.linalg.norm(original_info.t5_embeds, dim=-1)
            original_max_norm = torch.max(original_norms)

            new_norms = torch.linalg.norm(new_t5_embeds, dim=-1)
            new_max_norm = torch.max(new_norms)

            context.logger.info(f"Original max norm: {original_max_norm.item():.4f}")
            context.logger.info(f"Max norm after scaling section(s): {new_max_norm.item():.4f}")

            # Add a small epsilon to avoid division by zero
            epsilon = torch.finfo(new_t5_embeds.dtype).eps
            if new_max_norm > epsilon:
                rescale_factor = original_max_norm / new_max_norm
                new_t5_embeds = new_t5_embeds * rescale_factor
                context.logger.info(
                    f"Rescaling by a factor of {rescale_factor.item():.4f} to restore original max norm."
                )

                new_norms = torch.linalg.norm(new_t5_embeds, dim=-1)
                new_max_norm = torch.max(new_norms)

                context.logger.info(f"Max norm after Rescaling: {new_max_norm.item():.4f}")

        new_info = FLUXConditioningInfo(clip_embeds=original_info.clip_embeds.clone(), t5_embeds=new_t5_embeds)
        new_cond_data = ConditioningFieldData(conditionings=[new_info])
        new_cond_name = context.conditioning.save(new_cond_data)

        return FluxConditioningOutput(
            conditioning=FluxConditioningField(
                conditioning_name=new_cond_name,
                mask=self.conditioning.mask,
            )
        )

    def _process_embeds(
        self,
        context: InvocationContext,
        embeds: torch.Tensor,
        tokenizer_loader: ModelIdentifierField,
        prompt: str,
        sections: list[str],
        scales: list[float],
        positions: Optional[list[int]] = None,
    ) -> torch.Tensor:
        # Pooled embeddings (e.g. from CLIP) are 2D. Sequence embeddings (e.g. from T5) are 3D.
        # We can only scale a section of a sequence embedding.
        if embeds.dim() < 3:
            context.logger.warning(f"Cannot apply prompt section scaling to a {embeds.dim()}D tensor. Skipping.")
            return embeds.clone()

        multipliers = torch.ones(embeds.shape[1], device=embeds.device, dtype=embeds.dtype)

        with context.models.load(tokenizer_loader) as tokenizer:
            # Tokenize the prompt to get token offsets.
            assert isinstance(tokenizer, T5TokenizerFast)
            encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
            token_offsets = encoding["offset_mapping"]
            input_ids = encoding["input_ids"]

            if len(input_ids) > embeds.shape[1]:
                context.logger.warning(
                    f"The tokenized prompt length ({len(input_ids)}) exceeds the conditioning tensor "
                    f"length ({embeds.shape[1]}). The calculated token indices may be incorrect. This can happen "
                    "if the prompt was truncated during the original encoding."
                )

            if positions is not None:
                # If positions are provided, use them to locate the sections to scale.
                for i, start_char in enumerate(positions):
                    section = sections[i]
                    scale = scales[i]
                    end_char = start_char + len(section)

                    # Sanity check that the section text at the given position matches.
                    if not prompt.startswith(section, start_char):
                        context.logger.warning(
                            f"The prompt section '{section}' was not found at character position {start_char}. Skipping."
                        )
                        continue

                    token_indices = [
                        idx
                        for idx, (start, end) in enumerate(list(token_offsets))
                        if start < end_char and end > start_char
                    ]

                    if not token_indices:
                        context.logger.warning(
                            f"Could not find tokens for section '{section}' at position {start_char}."
                        )
                        continue

                    start_token_idx = min(token_indices)
                    end_token_idx = max(token_indices) + 1

                    if end_token_idx > multipliers.shape[0]:
                        context.logger.warning(
                            f"Section '{section}' at position {start_char} extends beyond conditioning length of {multipliers.shape[0]}. Truncating."
                        )
                        end_token_idx = multipliers.shape[0]

                    context.logger.info(
                        f"POS: Scaled '{section}' by {scale:.2f} at tokens {start_token_idx} to {end_token_idx - 1}."
                    )
                    multipliers[start_token_idx:end_token_idx] *= scale
            else:
                # If no positions, find all occurrences of sections in the prompt string.
                for i, section in enumerate(sections):
                    scale = scales[i]

                    if not section:
                        context.logger.warning(f"Prompt section at index {i} is empty, not scaling.")
                        continue

                    # Find all occurrences of the section string.
                    for match in re.finditer(re.escape(section), prompt):
                        start_char, end_char = match.span()

                        token_indices = [
                            idx
                            for idx, (start, end) in enumerate(list(token_offsets))
                            if start < end_char and end > start_char
                        ]

                        if not token_indices:
                            # This can happen if the section is part of a larger token, e.g. searching for "cat" in "caterpillar"
                            context.logger.warning(
                                f"Could not find tokens for section '{section}' at position {start_char}."
                            )
                            continue

                        start_token_idx = min(token_indices)
                        end_token_idx = max(token_indices) + 1

                        if end_token_idx > multipliers.shape[0]:
                            context.logger.warning(
                                f"Section '{section}' at position {start_char} extends beyond conditioning length of {multipliers.shape[0]}. Truncating."
                            )
                            end_token_idx = multipliers.shape[0]

                        context.logger.info(
                            f"Norm: Scaled '{section}' by {scale:.2f} at tokens {start_token_idx} to {end_token_idx - 1}."
                        )
                        multipliers[start_token_idx:end_token_idx] *= scale

        # Reshape multipliers to (1, sequence_length, 1) to broadcast correctly with embeds (1, sequence_length, embedding_dim)
        # and apply the scaling in a single vectorized operation.
        return embeds * multipliers.view(1, -1, 1)


@invocation_output("flux_weighted_prompt_output")
class FluxWeightedPromptOutput(BaseInvocationOutput):
    """Outputs a FLUX conditioning and a cleaned prompt"""

    conditioning: FluxConditioningField = OutputField(description="The FLUX conditioning")
    cleaned_prompt: str = OutputField(description="The prompt with all weighting syntax removed")


@invocation(
    "flux_weighted_prompt",
    title="FLUX Weighted Prompt",
    tags=["prompt", "conditioning", "flux", "weighted"],
    category="conditioning",
    version="1.0.0",
)
class FluxWeightedPromptInvocation(BaseInvocation):
    """Parses a weighted prompt, then encodes it for FLUX."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    t5_max_seq_len: Literal[256, 512] = InputField(
        description="Max sequence length for the T5 encoder. Expected to be 256 for FLUX schnell models and 512 for FLUX dev models."
    )
    mask: Optional[TensorField] = InputField(
        default=None, description="A mask defining the region that this conditioning prompt applies to."
    )
    rescale_output: bool = InputField(
        default=False,
        description="Rescales the output T5 embeddings to have the same max vector norm as the original conditioning.",
    )

    def _nested_parse_prompt(self, prompt: str) -> list[tuple[str, float]]:
        """
        Parses a prompt with potentially nested weights and returns a list of (text, weight) tuples.
        """
        parser = NestedPromptParser(prompt)
        ast = parser.parse()
        weighted_segments = flatten_ast(ast)

        if not weighted_segments:
            return []

        # The output of flatten_ast might have consecutive text segments with the same weight.
        # We can merge them for efficiency.
        merged_segments = [weighted_segments[0]]
        for text, weight in weighted_segments[1:]:
            last_text, last_weight = merged_segments[-1]
            if last_weight == weight:
                merged_segments[-1] = (last_text + text, weight)
            else:
                merged_segments.append((text, weight))

        return merged_segments

    def invoke(self, context: InvocationContext) -> FluxWeightedPromptOutput:
        # Parse the prompt into weighted segments
        segments = self._nested_parse_prompt(self.prompt)

        # join the texts to create a clean prompt without any weighting syntax
        texts = [text for text, _ in segments]
        cleaned_prompt = "".join(texts)

        # Encode the cleaned prompt using the existing invocation
        text_encoder = FluxTextEncoderInvocation(
            prompt=cleaned_prompt,
            clip=self.clip,
            t5_encoder=self.t5_encoder,
            t5_max_seq_len=self.t5_max_seq_len,
        )
        text_encoder_output = text_encoder.invoke(context)

        # Load the conditioning data
        cond_data = context.conditioning.load(text_encoder_output.conditioning.conditioning_name)
        original_info = cond_data.conditionings[0]
        assert isinstance(original_info, FLUXConditioningInfo)

        new_t5_embeds = original_info.t5_embeds.clone()

        # Apply scaling to the T5 embeddings based on the parsed segments
        with context.models.load(self.t5_encoder.tokenizer) as tokenizer:
            # Tokenize the cleaned prompt to get token offsets

            assert isinstance(tokenizer, T5TokenizerFast)
            encoding = tokenizer(cleaned_prompt, return_offsets_mapping=True, add_special_tokens=False)
            token_offsets = encoding["offset_mapping"]

            current_char_idx = 0
            for i in range(len(texts)):
                text = texts[i]
                weight = segments[i][1]  # get weight from original segments
                if weight != 1.0:
                    start_char = current_char_idx
                    end_char = start_char + len(text)

                    token_indices = [
                        idx for idx, (start, end) in enumerate(token_offsets) if start < end_char and end > start_char
                    ]

                    if token_indices:
                        start_token_idx = min(token_indices)
                        end_token_idx = max(token_indices) + 1

                        context.logger.info(
                            f"Weighted: Scaled '{text}' by {weight:.2f} at tokens {start_token_idx} to {end_token_idx - 1}."
                        )
                        new_t5_embeds[:, start_token_idx:end_token_idx, :] *= weight
                    else:
                        context.logger.warning(f"Could not find tokens for segment '{text}'.")

                current_char_idx += len(text)

        if self.rescale_output:
            original_norms = torch.linalg.norm(original_info.t5_embeds, dim=-1)
            original_max_norm = torch.max(original_norms)
            new_norms = torch.linalg.norm(new_t5_embeds, dim=-1)
            new_max_norm = torch.max(new_norms)
            context.logger.info(f"Original max norm: {original_max_norm.item():.4f}")
            context.logger.info(f"Max norm after scaling section(s): {new_max_norm.item():.4f}")
            epsilon = torch.finfo(new_t5_embeds.dtype).eps
            if new_max_norm > epsilon:
                rescale_factor = original_max_norm / new_max_norm
                new_t5_embeds = new_t5_embeds * rescale_factor
                context.logger.info(
                    f"Rescaling by a factor of {rescale_factor.item():.4f} to restore original max norm."
                )
                new_norms = torch.linalg.norm(new_t5_embeds, dim=-1)
                new_max_norm = torch.max(new_norms)
                context.logger.info(f"Max norm after Rescaling: {new_max_norm.item():.4f}")

        new_info = FLUXConditioningInfo(clip_embeds=original_info.clip_embeds.clone(), t5_embeds=new_t5_embeds)
        new_cond_data = ConditioningFieldData(conditionings=[new_info])
        new_cond_name = context.conditioning.save(new_cond_data)

        return FluxWeightedPromptOutput(
            conditioning=FluxConditioningField(conditioning_name=new_cond_name, mask=self.mask),
            cleaned_prompt=cleaned_prompt,
        )
