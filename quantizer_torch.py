from collections.abc import Callable
from functools import singledispatchmethod
from typing import Union, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn

from utils import strategy_dict

# Constants
log2 = torch.log(torch.tensor(2.0))


def q_round(x: torch.Tensor, strategy: int = 0) -> torch.Tensor:
    """Round the tensor.

    strategy:
        0: standard round (default, round to nearest, 0.5 to even)
        1: stochastic round
        2: fast uniform noise injection (uniform noise in [-0.5, 0.5])
        3: floor

    """
    if strategy == 0:  # standard round
        return torch.floor(x + 0.5)
    if strategy == 1:  # stochastic round
        _floor = torch.floor(x)
        noise = torch.rand_like(x)
        return torch.where(noise < x - _floor, _floor + 1, _floor)
    if strategy == 2:  # fast uniform noise injection
        noise = torch.rand_like(x) - 0.5  # uniform in [-0.5, 0.5]
        noise = torch.where(torch.abs(x) <= 0.5, -x, noise)
        return noise.detach() + x
    if strategy == 3:
        return torch.floor(x)
    raise ValueError(f"Unknown strategy {strategy}")


def get_arr_bits(arr: np.ndarray):
    """Internal helper function to compute the position of the highest and
    lowest bit of an array of fixed-point integers."""
    kn = (arr < 0).astype(np.int8)
    mul = 32
    arr = arr * 2**mul
    arr = np.abs(arr)[..., None]
    n = int(np.ceil(np.max(np.log2(arr + 1))))
    divisor = 2**np.arange(1, n)[None, ...]
    low_pos = np.sum(arr % divisor == 0, axis=-1) + (arr[..., 0] == 1)
    with np.errstate(divide='ignore'):
        high_pos = np.where(arr[..., 0] != 0, np.floor(np.log2(arr[..., 0]) + 1), low_pos).astype(np.int8)
    fb = 32 - low_pos
    int_bits = high_pos - low_pos - fb
    zero_mask = int_bits + fb == 0
    int_bits[zero_mask] = 0
    fb[zero_mask] = 0
    return kn.astype(np.int8), int_bits.astype(np.int8), fb.astype(np.int8)


class HGQ(nn.Module):
    """Heterogenous quantizer."""

    def __init__(self, 
                 init_bw: float, 
                 skip_dims, 
                 rnd_strategy: Union[str, int] = 'floor', 
                 exact_q_value: bool = True, 
                 dtype=None, 
                 bw_clip: Tuple[int, int] = (-23, 23), 
                 trainable: bool = True, 
                 regularizer: Optional[Callable] = None, 
                 minmax_record: bool = False):
        super().__init__()
        
        self.init_bw = init_bw
        self.skip_dims = skip_dims
        """tuple: Dimensions to use uniform quantizer. If None, use full heterogenous quantizer."""
        self.rnd_strategy = strategy_dict.get(rnd_strategy, -1) if isinstance(rnd_strategy, str) else rnd_strategy
        """How to round the quantized value. 0: standard round (default, round to nearest, round-up 0.5), 1: stochastic round, 2: fast uniform noise injection (uniform noise in [-0.5, 0.5]), 3: floor"""
        self.exact_q_value = exact_q_value
        """bool: Whether to use exact quantized value during training."""
        self.dtype = dtype or torch.float32
        self.bw_clip = bw_clip
        """tuple: (min, max) of bw. 23 by default in favor of float32 mantissa."""
        self.trainable = trainable
        self.regularizer = regularizer
        """Regularizer for bw."""
        self.minmax_record = minmax_record
        """bool: Whether to record min and max of quantized values."""
        self.built = False
        self.degeneracy = 1.
        """Degeneracy of the quantizer. Records how many values are mapped to the same quantizer."""
        
        # Initialize parameters that will be set during build
        self.fbw = None
        self._min = None
        self._max = None

    def _compute_bw_shape_and_degeneracy(self, input_shape):
        """Map skip_dims to input_shape and compute degeneracy."""
        if isinstance(self.skip_dims, str):
            if self.skip_dims == 'all':
                self.skip_dims = tuple(range(len(input_shape)))
            elif self.skip_dims == 'batch':
                self.skip_dims = (0,)
            elif self.skip_dims == 'none':
                self.skip_dims = None
            elif self.skip_dims == 'except_last':
                self.skip_dims = tuple(range(len(input_shape) - 1))
            elif self.skip_dims == 'except_1st':
                self.skip_dims = (0,) + tuple(range(2, len(input_shape)))
            else:
                raise ValueError("skip_dims must be tuple or str in ['all', 'except_last', 'batch', 'except_last', 'except_1st', 'none']")
        _input_shape = list(input_shape)
        degeneracy = 1
        if self.skip_dims:
            for d in self.skip_dims:
                degeneracy *= int(_input_shape[d]) if _input_shape[d] is not None else 1
                _input_shape[d] = 1
        return _input_shape, degeneracy

    def init_minmax(self):
        if self.minmax_record:
            self._min = nn.Parameter(torch.zeros_like(self.fbw), requires_grad=False)
            self._max = nn.Parameter(torch.zeros_like(self.fbw), requires_grad=False)
            self.minmax_reg_reset()

    @singledispatchmethod
    def build(self, fbw, name=None):
        self.built = True
        self.fbw = fbw
        self.init_minmax()

    @build.register
    def build_fn_for_tuple_fbw(self, input_shape: tuple, name: Optional[str] = None):
        self.built = True
        _input_shape, degeneracy = self._compute_bw_shape_and_degeneracy(input_shape)
        self.degeneracy = degeneracy
        # initialize the bitwidth tensor
        fbw_tensor = torch.ones(_input_shape, dtype=self.dtype) * self.init_bw
        if self.trainable:
            self.fbw = nn.Parameter(fbw_tensor)
        else:
            self.register_buffer('fbw', fbw_tensor)

        self.init_minmax()

    def minmax_reg_reset(self):
        """Reset min and max to inf and -inf, respectively."""
        assert self.built
        inf = torch.full_like(self.fbw, float('inf'))
        with torch.no_grad():
            self._min.copy_(inf)
            self._max.copy_(-inf)

    def forward(self, x: torch.Tensor, training: Optional[bool] = None, record_minmax: Optional[bool] = None) -> torch.Tensor:
        if not self.built:
            self.build(tuple(x.shape), name=None)
        
        # Use self.training if training is not explicitly provided
        if training is None:
            training = self.training
            
        if self.bw_clip:
            with torch.no_grad():
                self.fbw.clamp_(*self.bw_clip)
        
        return self._forward_impl(x, training, record_minmax)

    def _forward_impl(self, x: torch.Tensor, training: bool, record_minmax: Optional[bool]) -> torch.Tensor:
        """Forward pass of HGQ.
        Args:
            training: if set to True, gradient will be propagated through the quantization process.
            record_minmax: if set to True, min and max of quantized values will
            be recorded for deriving the necessary integer bits. Only necessary
            for activation/pre-activation values.
        """
        if self.exact_q_value or not training:
            scale = torch.pow(torch.tensor(2.0), torch.round(self.fbw))
        else:
            scale = torch.pow(torch.tensor(2.0), self.fbw)

        rnd_strategy = self.rnd_strategy
        if not training and rnd_strategy != 3:  # not std round or floor
            rnd_strategy = 0
        
        xq = q_round(x * scale, rnd_strategy) / scale
        
        # we cannot conpute gradient for xq - x
        delta = (xq - x).detach() # = stop_gradient in tf
        if training:
            prod = delta * self.fbw * log2 # calculate an approximate gradient
            delta = (delta + prod).detach() - prod

        if not record_minmax:
            return x + delta
            
        xq = x + delta
        if self.skip_dims:
            min_xq = torch.amin(xq, dim=self.skip_dims, keepdim=True)
            max_xq = torch.amax(xq, dim=self.skip_dims, keepdim=True)
        else:
            min_xq = max_xq = xq

        with torch.no_grad():
            self._min.copy_(torch.minimum(min_xq, self._min))
            self._max.copy_(torch.maximum(max_xq, self._max))

        return xq

    def bias_forward(self, x: torch.Tensor, training: Optional[bool] = None, channel_loc: int = -1) -> torch.Tensor:
        """Forward pass for the bias term. Grammatical sugar"""
        if training is None:
            training = self.training
            
        if channel_loc == -1:
            dims = list(range(len(self.fbw.shape) - 1))
        elif channel_loc == 1:
            dims = [0] + list(range(2, len(self.fbw.shape)))
        else:
            raise ValueError('channel_loc must be -1 or 1')

        fbw = torch.amax(self.fbw, dim=dims, keepdim=False)

        if self.exact_q_value or not training:
            scale = torch.pow(torch.tensor(2.0), torch.round(fbw))
        else:
            scale = torch.pow(torch.tensor(2.0), fbw)

        rnd_strategy = self.rnd_strategy
        if not training and rnd_strategy != 3:  # not std round or floor
            rnd_strategy = 0
            
        xq = q_round(x * scale, rnd_strategy) / scale

        delta = (xq - x).detach()
        if training:
            prod = delta * fbw * log2
            delta = (delta + prod).detach() - prod

        return x + delta

    def get_bits(self, ref: Optional[torch.Tensor] = None, quantized: Optional[bool] = None, pos_only: bool = False):
        """Get approximated int/frac/keep_negative bits of the equivalent fixed-point quantizer.
        Args:
            ref: Input tensor to compute the bits. If None, use the min/max record.
            quantized: If input is already quantized. Skip quantization pass if set to True.
            pos_only: If True, only compute the bits for positive values. Useful if have a ReLU layer after.
        """
        fp_bits = torch.round(self.fbw)
        fp_bits = self.fbw + (fp_bits - self.fbw).detach()
        
        if ref is not None:
            if quantized:
                _ref = ref
            else:
                _ref = self._forward_impl(ref, training=False, record_minmax=False)
            kn = (_ref < 0).float()
            _ref = torch.abs(_ref)
            int_bits = torch.floor(torch.log(_ref) / log2) + 1
            if self.skip_dims:
                int_bits = torch.amax(int_bits, dim=self.skip_dims, keepdim=True)
                kn = torch.amax(kn, dim=self.skip_dims, keepdim=True)
        else:
            assert self.minmax_record
            if pos_only:
                _ref = torch.maximum(self._max, torch.tensor(0.0))
                kn = torch.zeros_like(self._max)
            else:
                _ref = torch.maximum(torch.abs(self._min), torch.abs(self._max))
                kn = (self._min < 0).float()
            int_bits = torch.floor(torch.log(_ref) / log2) + 1
        return int_bits, fp_bits, kn

    def get_bits_exact(self, ref: Optional[torch.Tensor] = None, pos_only: bool = False):
        """Get exact int/frac/keep_negative bits of the equivalent fixed-point quantizer.
        Args:
            ref: Input tensor to compute the bits. If None, use the min/max record.
            pos_only: If True, only compute the bits for positive values. Useful if have a ReLU layer after.
        """
        if ref is None and self.minmax_record:
            assert torch.all(self._max - self._min >= 0), "Some min values are larger than max values. Did you forget to run trace_minmax?"
            f = torch.round(self.fbw).detach().cpu().numpy()
            with np.errstate(divide='ignore'):
                if pos_only:
                    _ref = np.maximum(self._max.detach().cpu().numpy(), 0.)
                    i = np.floor(np.log2(_ref)) + 1
                    k = np.zeros_like(_ref)
                else:
                    min_vals = self._min.detach().cpu().numpy()
                    max_vals = self._max.detach().cpu().numpy()
                    i_neg = np.ceil(np.log2(np.abs(min_vals)))
                    i_pos = np.ceil(np.log2(np.abs(max_vals + 2.**-f)))
                    i = np.maximum(i_neg, i_pos)
                    k = (min_vals < 0)
            i = np.clip(i, -f - k, 32)
            k, i, f = k.astype(np.int8), i.astype(np.int8), f.astype(np.int8)
            mask = k + i + f != 0
            return k * mask, i * mask, f * mask

        assert ref is not None
        w = self._forward_impl(ref, training=False, record_minmax=False).detach().cpu().numpy()
        k, i, f = get_arr_bits(w)
        mask = k + i + f != 0
        return k * mask, i * mask, f * mask

    def adapt_bw_bits(self, ref: torch.Tensor):
        """Adapt the bitwidth of the quantizer to the input tensor,
        such that each input is represented with approximately the same number of bits.
        (1.5 with be represented by ap_fixed<2,1>
        and 0.375 will be represented by ap_fixed<2,-2>).

        It shifts bits between integer and fractional parts
        to keep total bits fixed and accuracy high.
        Big numbers → more integer bits, fewer fractional bits
        Small numbers → fewer integer bits, more fractional bits"""
        if not self.built:
            self.build(tuple(ref.shape), name=None)
        new_fbw = self.fbw - (torch.log(torch.abs(ref)) / log2)
        if self.skip_dims:
            new_fbw = torch.amin(new_fbw, dim=self.skip_dims, keepdim=True)
        with torch.no_grad():
            self.fbw.copy_(new_fbw)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)