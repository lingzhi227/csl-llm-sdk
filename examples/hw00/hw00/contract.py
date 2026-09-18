"""Explicit single-request text inference state contract for sequential CS-3 jobs."""
from dataclasses import asdict, dataclass
from math import prod

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYERS = ((0, 20), (20, 44), (44, 64))  # Half-open; embedding in 0, norm/head in 2.


@dataclass(frozen=True)
class Geometry:
    hidden: int = 5120
    vocab: int = 248320
    kv_heads: int = 4
    head_dim: int = 256
    linear_key_heads: int = 16
    linear_value_heads: int = 48
    key_dim: int = 128
    value_dim: int = 128
    conv_width: int = 4
    context_limit: int = 2048
    profile: str = "qwen38_text_production_contract"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "profile" and (type(value) is not int or value <= 0):
                raise ValueError("Positive integer geometry required")

    @property
    def conv_channels(self):
        return 2 * self.linear_key_heads * self.key_dim + self.linear_value_heads * self.value_dim


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple

    @property
    def nbytes(self):
        return prod(self.shape) * {"bf16_le": 2, "f32_le": 4}[self.dtype]

    def metadata(self):
        return {"dtype": self.dtype, "shape": list(self.shape), "bytes": self.nbytes}


def output_schema(stage, end_position, token_count, geometry=Geometry()):
    """Packed raw tensors; batch size one is omitted. All cache positions are valid.

    K is post-normalization/RoPE; V is the cache value. Recurrent state axes are
    [value_head,key,value]. Convolution slots are chronological oldest to newest,
    zero-left-padded before four tokens. Backend adapters must transpose/rotate
    their physical layout to this canonical representation on export/import.
    """
    if (type(stage) is not int or stage not in range(3)
            or type(end_position) is not int or type(token_count) is not int
            or not 0 < token_count <= end_position <= geometry.context_limit):
        raise ValueError("Invalid stage or position extent")
    result = {}
    for layer in range(*LAYERS[stage]):
        prefix = f"layer_{layer:02d}"
        if layer % 4 == 3:
            for role in ("key", "value"):
                result[f"{prefix}_{role}"] = Tensor("bf16_le", (geometry.kv_heads, end_position, geometry.head_dim))
        else:
            result[f"{prefix}_recurrent"] = Tensor("f32_le", (geometry.linear_value_heads, geometry.key_dim, geometry.value_dim))
            result[f"{prefix}_conv"] = Tensor("bf16_le", (geometry.conv_channels, geometry.conv_width))
    result["hidden"] = Tensor("bf16_le", (token_count, geometry.hidden))
    if stage == 2:
        # Complete vocabulary at the last input position; greedy token is checked
        # against all logits, never a selected vocabulary subset.
        result["last_logits"] = Tensor("f32_le", (geometry.vocab,))
    return result
