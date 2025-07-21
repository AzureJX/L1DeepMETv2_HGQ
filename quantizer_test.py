from quantizer import HGQ
import numpy as np
import tensorflow as tf

# Example 1: Basic tensor quantization
def basic_tensor_quantization():
    """
    Simple example of quantizing a tensor using HGQ
    """
    print("=== Basic Tensor Quantization ===")
    
    # Create a sample tensor
    x = tf.random.normal((4, 8, 8, 16), dtype=tf.float32)
    print(f"Original tensor shape: {x.shape}")
    print(f"Original tensor range: [{tf.reduce_min(x):.4f}, {tf.reduce_max(x):.4f}]")
    
    # Create HGQ quantizer
    # Parameters:
    # - init_bw: initial bitwidth (8 bits)
    # - skip_dims: dimensions to skip for heterogeneous quantization
    # - rnd_strategy: rounding strategy (0=standard, 1=stochastic, 2=noise, 3=floor)
    quantizer = HGQ(
        init_bw=8.0,
        skip_dims=None,  # Use full heterogeneous quantization
        rnd_strategy=0,  # Standard rounding
        exact_q_value=True,
        trainable=True,
        minmax_record=False
    )
    
    # Build the quantizer with tensor shape
    quantizer.build(tuple(x.shape), name='basic_quantizer')
    
    # Quantize the tensor
    x_quantized = quantizer(x, training=True)
    
    print(f"Quantized tensor shape: {x_quantized.shape}")
    print(f"Quantized tensor range: [{tf.reduce_min(x_quantized):.4f}, {tf.reduce_max(x_quantized):.4f}]")
    
    # Expected results:
    # - Shape: Should be identical to input (4, 8, 8, 16)
    # - Range: Should be similar to input but with discrete quantized values
    # - With 8-bit quantization, values will be quantized to 2^8 = 256 levels
    
    # Check quantization error
    error = tf.reduce_mean(tf.abs(x - x_quantized))
    print(f"Mean absolute quantization error: {error:.6f}")
    
    return x_quantized


# Example 2: Different skip_dims configurations
def skip_dims_examples():
    """
    Demonstrate different skip_dims configurations
    """
    print("\n=== Skip Dims Examples ===")
    
    x = tf.random.normal((2, 4, 4, 8), dtype=tf.float32)
    
    configs = [
        ('none', None),
        ('all', 'all'),
        ('batch', 'batch'),
        ('except_last', 'except_last'),
        ('except_1st', 'except_1st'),
        ('custom', (0, 1))  # Skip batch and height dimensions
    ]
    
    for name, skip_dim in configs:
        print(f"\nConfiguration: {name} (skip_dims={skip_dim})")
        
        quantizer = HGQ(
            init_bw=6.0,
            skip_dims=skip_dim,
            rnd_strategy=0
        )
        
        quantizer.build(tuple(x.shape), name=f'quantizer_{name}')
        
        print(f"Bitwidth shape: {quantizer.fbw.shape}")
        print(f"Degeneracy: {quantizer.degeneracy}")
        
        x_quantized = quantizer(x, training=True)
        error = tf.reduce_mean(tf.abs(x - x_quantized))
        print(f"Quantization error: {error:.6f}")
        
        # Expected results:
        # Small values: Adapted bitwidth will be lower (more fractional bits)
        # Large values: Adapted bitwidth will be higher (more integer bits)
        # The adaptation should result in similar relative quantization accuracy
        
        # Expected results for each configuration:
        if name == 'none':
            # Shape: (2, 4, 4, 8) - same as input, each element has its own bitwidth
            # Degeneracy: 1 (no shared quantization)
            print("  Expected: Full heterogeneous quantization, lowest error")
        elif name == 'all':
            # Shape: (1, 1, 1, 1) - single bitwidth for entire tensor
            # Degeneracy: 2*4*4*8 = 256 (all elements share same quantization)
            print("  Expected: Single bitwidth, highest degeneracy")
        elif name == 'batch':
            # Shape: (1, 4, 4, 8) - same bitwidth across batch dimension
            # Degeneracy: 2 (batch dimension shared)
            print("  Expected: Shared across batch, degeneracy = 2")
        elif name == 'except_last':
            # Shape: (1, 1, 1, 8) - same bitwidth except last dimension
            # Degeneracy: 2*4*4 = 32
            print("  Expected: Channel-wise quantization, degeneracy = 32")
        elif name == 'except_1st':
            # Shape: (1, 4, 1, 1) - same bitwidth except first dimension
            # Degeneracy: 2*4*8 = 64
            print("  Expected: Skip batch and spatial dims, degeneracy = 64")


# Example 3: Different rounding strategies
def rounding_strategies_example():
    """
    Compare different rounding strategies
    """
    print("\n=== Rounding Strategies Comparison ===")
    
    x = tf.random.normal((1, 10, 10, 4), dtype=tf.float32)
    
    strategies = [
        (0, "Standard round"),
        (1, "Stochastic round"),
        (2, "Uniform noise injection"),
        (3, "Floor")
    ]
    
    for strategy, name in strategies:
        print(f"\nRounding strategy: {name}")
        
        quantizer = HGQ(
            init_bw=4.0,
            skip_dims='except_last',
            rnd_strategy=strategy
        )
        
        quantizer.build(tuple(x.shape), name=f'quantizer_rnd_{strategy}')
        
        # Quantize multiple times to see variance (especially for stochastic)
        results = []
        for _ in range(3):
            x_q = quantizer(x, training=True)
            error = tf.reduce_mean(tf.abs(x - x_q))
            results.append(error.numpy())
        
        print(f"Quantization errors: {[f'{e:.6f}' for e in results]}")
        print(f"Mean error: {np.mean(results):.6f}, Std: {np.std(results):.6f}")
        
        # Expected behavior for each strategy:
        if strategy == 0:
            print("  Expected: Consistent results (deterministic rounding)")
        elif strategy == 1:
            print("  Expected: Variable results (stochastic rounding introduces randomness)")
        elif strategy == 2:
            print("  Expected: Variable results (uniform noise injection)")
        elif strategy == 3:
            print("  Expected: Consistent results (deterministic floor operation)")


# Example 4: Adaptive bitwidth quantization
def adaptive_bitwidth_example():
    """
    Demonstrate adaptive bitwidth based on tensor values
    """
    print("\n=== Adaptive Bitwidth Example ===")
    
    # Create tensors with different dynamic ranges
    x_small = tf.random.normal((1, 8, 8, 4), stddev=0.1)  # Small values
    x_large = tf.random.normal((1, 8, 8, 4), stddev=10.0)  # Large values
    
    for i, (x, name) in enumerate([(x_small, "small"), (x_large, "large")]):
        print(f"\nTensor with {name} values:")
        print(f"Range: [{tf.reduce_min(x):.4f}, {tf.reduce_max(x):.4f}]")
        
        quantizer = HGQ(
            init_bw=8.0,
            skip_dims='except_last',
            rnd_strategy=0
        )
        
        quantizer.build(tuple(x.shape), name=f'adaptive_{i}')
        
        print(f"Initial bitwidth: {tf.reduce_mean(quantizer.fbw):.2f}")
        
        # Adapt bitwidth to tensor
        quantizer.adapt_bw_bits(x)
        
        print(f"Adapted bitwidth: {tf.reduce_mean(quantizer.fbw):.2f}")
        
        x_quantized = quantizer(x, training=True)
        error = tf.reduce_mean(tf.abs(x - x_quantized))
        print(f"Quantization error: {error:.6f}")


# Example 5: Min-max recording for activation quantization
def minmax_recording_example():
    """
    Demonstrate min-max recording for activation quantization
    """
    print("\n=== Min-Max Recording Example ===")
    
    # Create quantizer with min-max recording
    quantizer = HGQ(
        init_bw=6.0,
        skip_dims='except_last',
        rnd_strategy=0,
        minmax_record=True
    )
    
    # Build quantizer
    x_sample = tf.random.normal((1, 4, 4, 8))
    quantizer.build(tuple(x_sample.shape), name='minmax_quantizer')
    
    print("Recording min-max values from multiple batches...")
    
    # Process multiple batches to record min-max
    for i in range(5):
        x_batch = tf.random.normal((2, 4, 4, 8), stddev=1.0 + i * 0.5)
        _ = quantizer(x_batch, training=True, record_minmax=True)
        
        print(f"Batch {i+1}: min={tf.reduce_min(quantizer._min):.4f}, "
              f"max={tf.reduce_max(quantizer._max):.4f}")
    
    # Get bits based on recorded min-max
    int_bits, fp_bits, kn = quantizer.get_bits()
    total_bits = int_bits + fp_bits + kn
    
    print(f"Computed bits - Int: {tf.reduce_mean(int_bits):.2f}, "
          f"Frac: {tf.reduce_mean(fp_bits):.2f}, "
          f"Keep_neg: {tf.reduce_mean(kn):.2f}")
    print(f"Total effective bits: {tf.reduce_mean(total_bits):.2f}")
    
    # Get exact bits (non-differentiable)
    kn_exact, int_exact, fp_exact = quantizer.get_bits_exact()
    print(f"Exact bits - Int: {np.mean(int_exact):.2f}, "
          f"Frac: {np.mean(fp_exact):.2f}, "
          f"Keep_neg: {np.mean(kn_exact):.2f}")
    
    # Expected results:
    # - Min/max values should expand as more batches are processed
    # - Integer bits should reflect the maximum absolute values seen
    # - Fractional bits remain as configured (6.0)
    # - Keep_neg should be 1.0 if any negative values were encountered

if __name__ == "__main__":
    # basic_tensor_quantization()
    # skip_dims_examples()
    # rounding_strategies_example()
    # adaptive_bitwidth_example()
    minmax_recording_example()