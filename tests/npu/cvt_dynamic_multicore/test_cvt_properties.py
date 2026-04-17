"""Property-based tests for type conversion (cvt) operations.

These tests verify mathematical properties of type conversions using reference
implementations (PyTorch). They complement the hardware-specific NPU tests
by providing exhaustive property coverage.

Following the property-based testing skill workflow:
1. Detect Pattern: Type conversion (encode/decode-like)
2. Check Library: Hypothesis
3. Design Properties: Roundtrip, Bounds, Length preservation, Determinism
4. Create Strategies: Constrained by dtype ranges
5. Generate Tests: With edge cases
"""
from hypothesis import given, strategies as st, settings, example, assume
from hypothesis.extra.numpy import arrays
import pytest
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

_FLOAT_DTYPES = {"float32", "float16"}
_INT_DTYPES = {"int64", "int32", "int16", "int8", "uint8"}


def _iinfo(dtype_str: str):
    return torch.iinfo(_TORCH_DTYPE[dtype_str])


def _finfo(dtype_str: str):
    return torch.finfo(_TORCH_DTYPE[dtype_str])


def _cvt_reference(src: torch.Tensor, dst_dtype: str, rmode: str = None) -> torch.Tensor:
    """Reference implementation for type conversion."""
    dst_torch = _TORCH_DTYPE[dst_dtype]

    if dst_dtype in _INT_DTYPES:
        info = _iinfo(dst_dtype)
        src_f = src.float()
        if rmode == "cast_rint":
            rounded = torch.round(src_f)
        else:
            rounded = torch.trunc(src_f) if src.dtype.is_floating_point else src_f
        return rounded.clamp(info.min, info.max).to(dst_torch)

    return src.to(dst_torch)


@st.composite
def float32_tensors(draw, min_size: int = 1, max_size: int = 256):
    """Generate float32 tensors with values in safe range for conversion."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(
        arrays(
            dtype=np.float32,
            shape=(size,),
            elements=st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def float16_tensors(draw, min_size: int = 1, max_size: int = 256):
    """Generate float16 tensors with values safe for int8 conversion."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(
        arrays(
            dtype=np.float16,
            shape=(size,),
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def int32_tensors(draw, min_size: int = 1, max_size: int = 256):
    """Generate int32 tensors with values in int16 range."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(
        arrays(
            dtype=np.int32,
            shape=(size,),
            elements=st.integers(min_value=-30000, max_value=30000),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def int8_tensors(draw, min_size: int = 1, max_size: int = 256):
    """Generate int8 tensors."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(
        arrays(
            dtype=np.int8,
            shape=(size,),
            elements=st.integers(min_value=-128, max_value=127),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def uint8_tensors(draw, min_size: int = 1, max_size: int = 256):
    """Generate uint8 tensors."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(
        arrays(
            dtype=np.uint8,
            shape=(size,),
            elements=st.integers(min_value=0, max_value=255),
        )
    )
    return torch.from_numpy(arr)


class TestCvtFloat32ToFloat16Properties:
    """Property-based tests for float32 → float16 conversion (narrowing)."""

    @given(float32_tensors())
    @example(torch.tensor([0.0]))
    @example(torch.tensor([1.0]))
    @example(torch.tensor([-1.0]))
    @example(torch.tensor([65504.0]))
    @example(torch.tensor([-65504.0]))
    @settings(max_examples=50, deadline=None)
    def test_length_preserved(self, x: torch.Tensor):
        """Output has same length as input."""
        result = _cvt_reference(x, "float16")
        assert result.shape == x.shape, "Length not preserved"

    @given(float32_tensors())
    @example(torch.tensor([0.0]))
    @example(torch.tensor([1.0]))
    @settings(max_examples=50, deadline=None)
    def test_output_is_float16(self, x: torch.Tensor):
        """Output dtype is float16."""
        result = _cvt_reference(x, "float16")
        assert result.dtype == torch.float16, "Output is not float16"

    @given(float32_tensors())
    @example(torch.tensor([0.0]))
    @example(torch.tensor([1.0]))
    @settings(max_examples=50, deadline=None)
    def test_deterministic(self, x: torch.Tensor):
        """Same input always produces same output."""
        result1 = _cvt_reference(x, "float16")
        result2 = _cvt_reference(x, "float16")
        assert torch.equal(result1, result2), "Conversion is not deterministic"

    @given(float32_tensors())
    @example(torch.tensor([0.0]))
    @example(torch.tensor([1.0]))
    @settings(max_examples=50, deadline=None)
    def test_within_float16_range(self, x: torch.Tensor):
        """Output values are within float16 representable range."""
        result = _cvt_reference(x, "float16")
        finfo = _finfo("float16")
        assert torch.all(torch.abs(result) <= finfo.max), "Output exceeds float16 range"

    @given(float32_tensors())
    @example(torch.tensor([0.5]))
    @example(torch.tensor([0.25]))
    @settings(max_examples=50, deadline=None)
    def test_precision_bound(self, x: torch.Tensor):
        """Result is within 1 ULP of input (for representable values)."""
        result = _cvt_reference(x, "float16")
        finfo = _finfo("float16")
        tiny = finfo.smallest_normal
        eps = finfo.eps
        ulp = x.abs().clamp(min=tiny) * eps
        back_to_f32 = result.float()
        diff = (back_to_f32 - x).abs()
        assert torch.all(diff <= ulp + 1e-6), "Result exceeds 1 ULP bound"


class TestCvtFloat16ToInt8Properties:
    """Property-based tests for float16 → int8 conversion (cast_rint)."""

    @given(float16_tensors())
    @example(torch.tensor([0.0], dtype=torch.float16))
    @example(torch.tensor([1.0], dtype=torch.float16))
    @example(torch.tensor([-1.0], dtype=torch.float16))
    @example(torch.tensor([127.0], dtype=torch.float16))
    @example(torch.tensor([-128.0], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_length_preserved(self, x: torch.Tensor):
        """Output has same length as input."""
        result = _cvt_reference(x, "int8", rmode="cast_rint")
        assert result.shape == x.shape, "Length not preserved"

    @given(float16_tensors())
    @example(torch.tensor([0.0], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_output_is_int8(self, x: torch.Tensor):
        """Output dtype is int8."""
        result = _cvt_reference(x, "int8", rmode="cast_rint")
        assert result.dtype == torch.int8, "Output is not int8"

    @given(float16_tensors())
    @example(torch.tensor([0.0], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_within_int8_range(self, x: torch.Tensor):
        """Output values are within int8 range [-128, 127]."""
        result = _cvt_reference(x, "int8", rmode="cast_rint")
        info = _iinfo("int8")
        assert torch.all(result >= info.min), "Output below int8 min"
        assert torch.all(result <= info.max), "Output above int8 max"

    @given(float16_tensors())
    @example(torch.tensor([1.5], dtype=torch.float16))
    @example(torch.tensor([2.5], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_rounds_to_nearest(self, x: torch.Tensor):
        """Values are rounded to nearest integer (cast_rint semantics)."""
        result = _cvt_reference(x, "int8", rmode="cast_rint")
        expected_rounded = torch.round(x.float())
        info = _iinfo("int8")
        expected_clamped = expected_rounded.clamp(info.min, info.max)
        assert torch.all(result == expected_clamped.to(torch.int8)), \
            "Not rounding to nearest"

    @given(float16_tensors())
    @example(torch.tensor([0.0], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_deterministic(self, x: torch.Tensor):
        """Same input always produces same output."""
        result1 = _cvt_reference(x, "int8", rmode="cast_rint")
        result2 = _cvt_reference(x, "int8", rmode="cast_rint")
        assert torch.equal(result1, result2), "Conversion is not deterministic"


class TestCvtInt32ToInt16Properties:
    """Property-based tests for int32 → int16 conversion (narrowing)."""

    @given(int32_tensors())
    @example(torch.tensor([0], dtype=torch.int32))
    @example(torch.tensor([1], dtype=torch.int32))
    @example(torch.tensor([-1], dtype=torch.int32))
    @example(torch.tensor([32767], dtype=torch.int32))
    @example(torch.tensor([-32768], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_length_preserved(self, x: torch.Tensor):
        """Output has same length as input."""
        result = _cvt_reference(x, "int16")
        assert result.shape == x.shape, "Length not preserved"

    @given(int32_tensors())
    @example(torch.tensor([0], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_output_is_int16(self, x: torch.Tensor):
        """Output dtype is int16."""
        result = _cvt_reference(x, "int16")
        assert result.dtype == torch.int16, "Output is not int16"

    @given(int32_tensors())
    @example(torch.tensor([0], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_within_int16_range(self, x: torch.Tensor):
        """Output values are within int16 range [-32768, 32767]."""
        result = _cvt_reference(x, "int16")
        info = _iinfo("int16")
        assert torch.all(result >= info.min), "Output below int16 min"
        assert torch.all(result <= info.max), "Output above int16 max"

    @given(int32_tensors())
    @example(torch.tensor([0], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_deterministic(self, x: torch.Tensor):
        """Same input always produces same output."""
        result1 = _cvt_reference(x, "int16")
        result2 = _cvt_reference(x, "int16")
        assert torch.equal(result1, result2), "Conversion is not deterministic"

    @given(int32_tensors())
    @example(torch.tensor([100], dtype=torch.int32))
    @example(torch.tensor([-100], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_value_preserved_in_range(self, x: torch.Tensor):
        """Values within int16 range are preserved exactly."""
        assume(torch.all(x >= -32768) and torch.all(x <= 32767))
        result = _cvt_reference(x, "int16")
        assert torch.all(result == x), "Values in range not preserved"


class TestCvtRoundtripProperties:
    """Property-based tests for roundtrip conversions (widening then narrowing)."""

    @given(int8_tensors())
    @example(torch.tensor([0], dtype=torch.int8))
    @example(torch.tensor([127], dtype=torch.int8))
    @example(torch.tensor([-128], dtype=torch.int8))
    @settings(max_examples=50, deadline=None)
    def test_int8_to_int32_roundtrip(self, x: torch.Tensor):
        """int8 → int32 → int8 should preserve original values."""
        to_int32 = _cvt_reference(x, "int32")
        back_to_int8 = _cvt_reference(to_int32, "int8")
        assert torch.equal(back_to_int8, x), "Roundtrip failed for int8→int32→int8"

    @given(uint8_tensors())
    @example(torch.tensor([0], dtype=torch.uint8))
    @example(torch.tensor([255], dtype=torch.uint8))
    @settings(max_examples=50, deadline=None)
    def test_uint8_to_int32_roundtrip(self, x: torch.Tensor):
        """uint8 → int32 → uint8 should preserve original values."""
        to_int32 = _cvt_reference(x, "int32")
        back_to_uint8 = _cvt_reference(to_int32, "uint8")
        assert torch.equal(back_to_uint8, x), "Roundtrip failed for uint8→int32→uint8"


class TestCvtMonotonicityProperties:
    """Property-based tests for monotonicity of conversions."""

    @given(float32_tensors(min_size=2))
    @example(torch.tensor([0.0, 1.0]))
    @example(torch.tensor([-1.0, 0.0, 1.0]))
    @settings(max_examples=50, deadline=None)
    def test_float32_to_float16_monotonic(self, x: torch.Tensor):
        """If x[i] <= x[j], then cvt(x[i]) <= cvt(x[j]) for float32→float16."""
        sorted_x, _ = torch.sort(x)
        result = _cvt_reference(sorted_x, "float16")
        assert torch.all(result[:-1] <= result[1:]), \
            "float32→float16 conversion is not monotonic"

    @given(float16_tensors(min_size=2))
    @example(torch.tensor([0.0, 1.0], dtype=torch.float16))
    @settings(max_examples=50, deadline=None)
    def test_float16_to_int8_monotonic(self, x: torch.Tensor):
        """If x[i] <= x[j], then cvt(x[i]) <= cvt(x[j]) for float16→int8."""
        sorted_x, _ = torch.sort(x)
        result = _cvt_reference(sorted_x, "int8", rmode="cast_rint")
        assert torch.all(result[:-1] <= result[1:]), \
            "float16→int8 conversion is not monotonic"

    @given(int32_tensors(min_size=2))
    @example(torch.tensor([0, 1], dtype=torch.int32))
    @settings(max_examples=50, deadline=None)
    def test_int32_to_int16_monotonic(self, x: torch.Tensor):
        """If x[i] <= x[j], then cvt(x[i]) <= cvt(x[j]) for int32→int16."""
        sorted_x, _ = torch.sort(x)
        result = _cvt_reference(sorted_x, "int16")
        assert torch.all(result[:-1] <= result[1:]), \
            "int32→int16 conversion is not monotonic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--hypothesis-show-statistics"])
