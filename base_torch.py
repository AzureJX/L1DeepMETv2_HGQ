from collections.abc import Callable
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizer import HGQ
from utils import apf_to_tuple, get_default_kq_conf, get_default_paq_conf, tuple_to_apf, warn


"""
ABSBaseLayer
Lacks regularizer support
Incomplete activation handling
JIT compilation optimizations
Get_config?
Where is compute_output_shape defined?
"""

@torch.compile
def scale_grad(x, scale):
    sx = x * scale
    return sx + (x - sx).detach()

class ABSBaseLayer(nn.Module):
    """Abstract base layer with input tracking capabilities"""
    
    def __init__(self):
        super().__init__()
        self._input_layer = None
        self._is_reused = False
        self._usage_count = 0 # == len(self._inbound_nodes) in Keras
    
    """PyTorch does not track layer connections the way Keras does.
    call this function to manually set the input layer when building a layer"""
    def _track_input_layer(self, input_layer):
        """Track the input layer for this layer"""
        if self._input_layer is None:
            self._input_layer = input_layer
        elif self._input_layer != input_layer:
            self._is_reused = True # this layer is used multiple times
        self._usage_count += 1
    
    @property
    def last_layer(self):
        assert not self._is_reused, f'input_container is only available for layers used only once. {self.__class__.__name__} is used {self._usage_count} times.'
        return self._input_layer
    
    @property
    def input_bw(self):
        assert self._usage_count <= 1, f"Layer {self.__class__.__name__} is reused {self._usage_count} times. This is not allowed."
        try:
            return self.last_layer.act_bw
        except (AttributeError, AssertionError):
            return None
    
    @property
    def input_bw_exact(self):
        assert self._usage_count <= 1, f"Layer {self.__class__.__name__} is reused {self._usage_count} times. This is not allowed."
        return self.last_layer.act_bw_exact.float()


class HLayerBase(ABSBaseLayer):
    """Abstract base class for all layers in the library.
    Child classes should call post_build() after calling their build() method.
    """
    
    def __init__(self, kq_conf=None, paq_conf=None, beta=0., **kwargs):
        # super().__init__()        
        self._has_kernel = None
        self._has_bias = None
        self.kq_config = kq_conf or get_default_kq_conf()
        """kernel quantizer config"""
        self.paq_config = paq_conf or get_default_paq_conf()
        """pre-activation quantizer config"""
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=False)
        """BOPs-regularization strength"""
        self.record_minmax = False
        self._has_last_layer = False
        self._do_adapt_kernel_bits = kwargs.pop('do_adapt_kernel_bits', True)
        self._delayed_kernel_bits_adaption = False
        
        # Handle activation
        activation = kwargs.get('activation', None)
        if activation is not None and type(self).__name__ != 'HActivation':
            if isinstance(activation, str):
                activation = activation.lower()
            elif isinstance(activation, Callable):
                activation = activation.__name__.lower()
            assert activation in ['relu', 'linear'], f'activation other than relu and linear are only supported for HActivation layer, but is defined for {type(self).__name__} layer.'
        self.activation = activation

        self._built = False # Mimic the _built in Keras
        super().__init__(**kwargs) # Pass anything remained in kwargs to the parent
    
    def build(self, input_shape):
        """Build the layer with given input shape"""
        if self._built:
            return
        self.post_build(input_shape)
        self._built = True # == tf.keras.layers.Layer.build(input_shape)
    
    @property
    def can_bias_cover_rnd(self):
        if not self._has_bias:
            return False
        quantizer_shape = tuple(self.paq.fbw.shape)
        bias_shape = tuple(self.bias.shape)
        if len(bias_shape) != 1:
            warn(f'bias shape {bias_shape} is not supported.')
            return False
        if np.prod(quantizer_shape) == 1:
            return True
        if self.channel_loc == -1:
            return np.prod(quantizer_shape[:-1]) == 1 and bias_shape == quantizer_shape[-1:]
        elif self.channel_loc == 1:
            return np.prod(quantizer_shape[1:]) == 1 and bias_shape == quantizer_shape[0:1]
        return False
    
    @property
    def _relu_act(self):
        return hasattr(self, 'activation') and self.activation is F.relu
    
    def post_build(self, input_shape):
        """This method should be called after calling build() method of the
        child class. It initializes the quantizers and sets the bops variable,
        and set a few flags (_has_kernel, _has_bias, _relu_act) for convenience."""
        self._has_kernel = False
        self._has_bias = False
        if hasattr(self, 'kernel') and self.kernel is not None:
            self._has_kernel = True
        if hasattr(self, 'bias') and self.bias is not None:
            self._has_bias = True
        
        self.init_quantizers(input_shape)
        if not hasattr(self, 'channel_loc'):
            self.channel_loc = -1
        if self.paq_config['rnd_strategy'] == 'auto':
            self.paq.rnd_strategy = 0 if self.can_bias_cover_rnd else 3
        
        self.bops = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
    
    def init_quantizers(self, input_shape):
        """Initializes the High Granularity Quantizers for the kernel
        and the pre-activation values.
        This method is called by post_build() method."""
        
        if self._has_kernel:
            kq = HGQ.from_config(self.kq_config)
            bw_shape, degeneracy = kq._compute_bw_shape_and_degeneracy(self.kernel.shape)
            fbw = nn.Parameter(
                torch.full(bw_shape, kq.init_bw, dtype=torch.float32),
                requires_grad=kq.trainable and self.trainable,
            )
            """not sure if we need the regularizer in the future"""
            """regularizer handling"""
            self.register_parameter('kernel_bw', fbw) # == name='kernel_bw' in Keras
            kq.build(fbw)
            kq.degeneracy = degeneracy
            if self._do_adapt_kernel_bits and not self._delayed_kernel_bits_adaption:
                kq.adapt_bw_bits(self.kernel)
                self._do_adapt_kernel_bits = False
            self.kq = kq
        
        aq = HGQ.from_config(self.paq_config)
        output_shape = self.compute_output_shape(input_shape)
        self.output_shape = output_shape
        bw_shape, degeneracy = aq._compute_bw_shape_and_degeneracy(output_shape)
        fbw = nn.Parameter(
            torch.full(bw_shape, aq.init_bw, dtype=torch.float32),
            requires_grad=aq.trainable and self.trainable
        )
        self.register_parameter('activation_bw', fbw)
        """regularizer handling would need to be implemented separately in PyTorch!!!"""
        
        aq.build(fbw)
        aq.degeneracy = degeneracy
        self.paq = aq
    
    def forward(self, x, training=None, record_minmax=None): # call in Keras
        if not self._built:
            self.build(tuple(x.shape))
        
        if record_minmax is None:
            record_minmax = self.training or self.record_minmax
        
        # Convert to float32 if needed
        dtype = self.dtype or torch.float32
        x = x.to(dtype)
        
        return self._forward(x, training=training, record_minmax=record_minmax)
    
    def _forward(self, x, training=None, record_minmax=None): # forward i Keras
        raise NotImplementedError
    
    # def compute_output_shape(self, input_shape):
    #     """Compute output shape given input shape - must be implemented by subclasses"""
    #     raise NotImplementedError
    
    @property
    @torch.compile
    def kernel_bw(self):
        """Returns (over) approximated bitwidth of the kernel. Differentiable."""
        int_bits, fp_bits, kn = self.kq.get_bits(self.fused_kernel)
        k_bw = F.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, torch.sqrt(1. / self.kq.degeneracy))
        return k_bw.expand_as(self.kernel)
    
    @torch.compile
    def _kernel_bw(self, qk):
        """Returns (over) approximated bitwidth of the kernel. Differentiable. Takes the differentiable quantized kernel as input to avoid recomputing."""
        int_bits, fp_bits, kn = self.kq.get_bits(qk, quantized=True)
        k_bw = F.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, torch.sqrt(1. / self.kq.degeneracy))
        return k_bw.expand_as(qk)
    
    @property
    def kernel_bw_exact(self):
        """Returns exact bitwidth of the kernel. Non-differentiable. 
        For post-training use."""
        kn, int_bits, fb = self.kq.get_bits_exact(self.fused_kernel)
        return int_bits + fb  # sign not considered for kernel
    
    @property
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        int_bits, fp_bits, kn = self.paq.get_bits(pos_only=self._relu_act)
        bw = int_bits + fp_bits
        if not self._relu_act:
            bw = bw + kn
        bw = F.relu(bw)
        bw = scale_grad(bw, torch.sqrt(1. / self.paq.degeneracy))
        return bw.expand((1,) + self.output_shape[1:])
    
    @property
    def act_bw_exact(self) -> np.ndarray:
        """Returns the exact bitwidth of the pre-activation values. 
        Non-differentiable. For post-training use."""
        kn, int_bits, fb = self.paq.get_bits_exact(pos_only=self._relu_act)
        bw = int_bits + fb + kn
        return np.broadcast_to(bw, (1,) + self.output_shape[1:])
    
    @property
    def fused_bias(self):
        return self.bias
    
    @property
    def fused_kernel(self):
        return self.weight
    
    @property
    @torch.compile
    def fused_qkernel(self):
        """Returns the final, quantized kernel for deployment. non-differentiable, should not be used for training."""
        # Using torch.no_grad() because this should be non-differentiable like the original
        with torch.no_grad():
            return self.kq(self.fused_kernel, training=False)
    
    @property
    @torch.compile
    def fused_qbias(self):
        """Returns the final, quantized bias for deployment. non-differentiable, should not be used for training. When using rounding to nearest and the bias can cover the rounding error, bias is pre-biased to cover the rounding shift 2^-fbw, and then TRN can be used instead RND without any loss of accuracy."""
        bias = self.paq.bias_forward(self.fused_bias, False, self.channel_loc)
        
        fbw = self.paq.fbw
        # Reduce all dims except the channel dimension
        if self.channel_loc == -1:
            dims = tuple(range(len(fbw.shape) - 1))
        elif self.channel_loc == 1:
            dims = tuple([0] + list(range(2, len(fbw.shape))))
        else:
            raise ValueError('channel_loc must be -1 or 1')
        
        """implementation 1: amax support multiple dims, max doesn't"""
        fbw = torch.amax(self.paq.fbw, dim=dims, keepdim=False)
        fbw = fbw.expand_as(bias)
        mask = torch.amax(self.act_bw, dim=dims, keepdim=False) > 0

        """implementation 2: for loops"""
        # fbw_reduced = fbw
        # for dim in sorted(dims, reverse=True):
        #     fbw_reduced = torch.max(fbw_reduced, dim=dim, keepdim=False)[0]
        # fbw_reduced = fbw_reduced.expand_as(bias)
        
        # mask_tensor = self.act_bw
        # for dim in sorted(dims, reverse=True):
        #     mask_tensor = torch.max(mask_tensor, dim=dim, keepdim=False)[0]
        # mask = mask_tensor > 0
        
        if self.paq.rnd_strategy != 3 and self.can_bias_cover_rnd:
            bias = torch.pow(2., -torch.round(fbw) - 1) + bias
        
        return torch.where(mask, bias, torch.zeros_like(bias))
    
    def reset_minmax(self):
        """Resets the recorded minmax values for the pre-activation quantizer."""
        self.paq.minmax_reg_reset()
    
    @property
    def compute_exact_bops(self):
        """Computes the exact bops for the layer. Non-differentiable. For post-training use."""
        self.bops.data.fill_(0.)
        return np.float32(0.)
    
    #For model layer reproduction
    def get_config(self):
        """Get layer configuration"""
        config = dict(
            class_name=self.__class__.__name__,
            kq_conf=self.kq_config,
            paq_conf=self.paq_config,
            beta=float(self.beta.numpy()),
            activation=self.activation,
        )
        return config
    
    # def get_pytorch_config(self):
    #     """Get PyTorch-specific configuration"""
    #     config = {
    #         'activation': self.activation,
    #         'channel_loc': self.channel_loc,
    #         'record_minmax': self.record_minmax
    #     }
    #     return config

# class MyLayer(ABSBaseLayer)
    
if __name__ == '__main__':
    test = ABSBaseLayer()