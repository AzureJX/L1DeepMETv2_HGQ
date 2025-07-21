import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import torch
import torch.nn as nn
import numpy as np
from base_torch import ABSBaseLayer, HLayerBase

# Assuming your class is imported like this:
# from your_module import HLayerBase, ABSBaseLayer

class TestHLayerBase(unittest.TestCase):
    
    def setUp(self):
        """Set up common mocks and test data"""
        # Mock the external dependencies
        self.mock_hgq_patcher = patch('base_torch.HGQ')
        self.mock_hgq = self.mock_hgq_patcher.start()
        
        self.mock_utils_patcher = patch.multiple(
            'base_torch',
            get_default_kq_conf=Mock(return_value={'init_bw': 4, 'trainable': True}),
            get_default_paq_conf=Mock(return_value={'init_bw': 8, 'trainable': True, 'rnd_strategy': 'auto'})
        )
        self.mock_utils_patcher.start()
        
        # Create a concrete test class since HLayerBase is abstract
        class TestLayer(HLayerBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Add required attributes that would be set by child classes
                self.weight = nn.Parameter(torch.randn(10, 5))
                self.bias = nn.Parameter(torch.randn(10))
                
            def _forward(self, x, training=None, record_minmax=None):
                return x  # Simple pass-through for testing
            
            def compute_output_shape(self, input_shape):
                return (input_shape[0], 10)  # Mock output shape
        
        self.TestLayer = TestLayer
    
    def tearDown(self):
        """Clean up patches"""
        self.mock_hgq_patcher.stop()
        self.mock_utils_patcher.stop()

    def test_initialization_with_mocked_configs(self):
        """Test that initialization properly calls config functions"""
        layer = self.TestLayer()
        
        # Verify config functions were called
        self.assertEqual(layer.kq_config, {'init_bw': 4, 'trainable': True})
        self.assertEqual(layer.paq_config['init_bw'], 8)
        print("✓ test_initialization_with_mocked_configs PASSED")
        
    @patch('base_torch.torch.tensor')
    def test_beta_parameter_creation(self, mock_tensor):
        """Test beta parameter is created correctly"""
        mock_tensor.return_value = torch.tensor(0.5)
        
        layer = self.TestLayer(beta=0.5)
        
        # Verify torch.tensor was called with correct arguments
        mock_tensor.assert_called_with(0.5, dtype=torch.float32)
        print("✓ test_beta_parameter_creation PASSED")

    def test_quantizer_initialization(self):
        """Test quantizer initialization with mocked HGQ"""
        # Set up mock quantizer
        mock_quantizer = Mock()
        mock_quantizer._compute_bw_shape_and_degeneracy.return_value = ((4,), 1.0)
        self.mock_hgq.from_config.return_value = mock_quantizer
        
        layer = self.TestLayer()
        
        # Trigger build
        input_shape = (32, 5)
        layer.build(input_shape)
        
        # Verify quantizer was created and configured
        self.assertEqual(self.mock_hgq.from_config.call_count, 2)  # kq and paq
        mock_quantizer.build.assert_called()
        self.assertEqual(layer.kq, mock_quantizer)
        print("✓ test_quantizer_initialization PASSED")

    def test_forward_calls_build_automatically(self):
        """Test that forward() automatically calls build() if not built"""
        layer = self.TestLayer()
        
        # Mock the build method to track if it's called
        with patch.object(layer, 'build') as mock_build:
            x = torch.randn(32, 5)
            layer.forward(x)
            
            # Verify build was called with correct input shape
            mock_build.assert_called_once_with((32, 5))

        print("✓ test_forward_calls_build_automatically PASSED")

    def test_property_calculations_with_mocked_quantizers(self):
        """Test bitwidth property calculations"""
        # Create mock quantizers with predictable behavior
        mock_kq = Mock()
        mock_kq.get_bits.return_value = (torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([1.0]))
        mock_kq.degeneracy = 2.0
        
        layer = self.TestLayer()
        layer.kq = mock_kq
        
        # Test kernel_bw property
        with patch.object(layer, 'fused_kernel', torch.randn(10, 5)):
            bw = layer.kernel_bw
            
            # Verify the quantizer was called
            mock_kq.get_bits.assert_called_once()
            self.assertIsInstance(bw, torch.Tensor)
        
        print("✓ test_property_calculations_with_mocked_quantizers PASSED")

    def test_can_bias_cover_rnd_logic(self):
        """Test the bias covering logic with different configurations"""
        layer = self.TestLayer()
        layer._has_bias = True
        layer.channel_loc = -1
        
        # Mock the paq quantizer
        mock_paq = Mock()
        mock_paq.fbw.shape = (1, 1, 10)  # Shape that should allow bias covering
        layer.paq = mock_paq
        
        # Test the property
        result = layer.can_bias_cover_rnd
        self.assertTrue(result)
        print("✓ test_can_bias_cover_rnd_logic PASSED")

    @patch('base_torch.warn')
    def test_unsupported_bias_shape_warning(self, mock_warn):
        """Test that warnings are issued for unsupported bias shapes"""
        layer = self.TestLayer()
        layer._has_bias = True
        layer.bias = nn.Parameter(torch.randn(2, 5))  # 2D bias (unsupported)
        
        mock_paq = Mock()
        mock_paq.fbw.shape = (10,)
        layer.paq = mock_paq
        
        result = layer.can_bias_cover_rnd
        
        # Verify warning was called
        mock_warn.assert_called_once()
        self.assertFalse(result)
        print("✓ test_unsupported_bias_shape_warning PASSED")

    def test_input_tracking_functionality(self):
        """Test the ABSBaseLayer input tracking"""
        layer1 = self.TestLayer()
        layer2 = self.TestLayer()
        
        # Simulate layer connections
        layer2._track_input_layer(layer1)
        
        self.assertEqual(layer2.last_layer, layer1)
        self.assertEqual(layer2._usage_count, 1)
        self.assertFalse(layer2._is_reused)

    def test_layer_reuse_detection(self):
        """Test detection of layer reuse"""
        layer1 = self.TestLayer()
        layer2 = self.TestLayer()
        layer3 = self.TestLayer()
        
        # Use layer3 with different inputs (simulating reuse)
        layer3._track_input_layer(layer1)
        layer3._track_input_layer(layer2)
        
        self.assertTrue(layer3._is_reused)
        self.assertEqual(layer3._usage_count, 2)
        
        # Should raise assertion error when trying to access last_layer
        with self.assertRaises(AssertionError):
            _ = layer3.last_layer

    def test_config_serialization(self):
        """Test get_config method"""
        layer = self.TestLayer(beta=0.7, activation='relu')
        
        config = layer.get_config()
        
        expected_keys = ['class_name', 'kq_conf', 'paq_conf', 'beta', 'activation']
        for key in expected_keys:
            self.assertIn(key, config)
        
        self.assertEqual(config['class_name'], 'TestLayer')
        self.assertEqual(config['activation'], 'relu')

    def test_dtype_conversion_in_forward(self):
        """Test that forward() handles dtype conversion"""
        layer = self.TestLayer()
        
        # Create input with different dtype
        x = torch.randn(32, 5, dtype=torch.float64)
        
        with patch.object(layer, '_forward') as mock_forward:
            layer.forward(x)
            
            # Verify _forward was called with float32 tensor
            call_args = mock_forward.call_args[0]
            converted_x = call_args[0]
            self.assertEqual(converted_x.dtype, torch.float32)

    @patch('base_torch.F.relu')
    def test_activation_handling(self, mock_relu):
        """Test activation function handling"""
        # Test with relu activation
        layer = self.TestLayer(activation='relu')
        self.assertEqual(layer.activation, 'relu')
        
        # Test with callable activation
        layer2 = self.TestLayer(activation=torch.relu)
        self.assertEqual(layer2.activation, 'relu')

    def test_invalid_activation_raises_error(self):
        """Test that invalid activations raise appropriate errors"""
        with self.assertRaises(AssertionError):
            layer = self.TestLayer(activation='sigmoid')  # Should fail for non-HActivation layer

    def test_exact_bops_computation(self):
        """Test exact BOPS computation"""
        layer = self.TestLayer()
        layer.bops = nn.Parameter(torch.tensor(5.0))  # Start with non-zero value
        
        result = layer.compute_exact_bops
        
        # Should reset bops to 0 and return 0
        self.assertEqual(layer.bops.item(), 0.0)
        self.assertEqual(result, np.float32(0.0))


class TestMockingStrategies(unittest.TestCase):
    """Examples of different mocking strategies"""
    
    def test_patch_as_decorator(self):
        """Using patch as a decorator"""
        @patch('base_torch.get_default_kq_conf')
        def test_method(mock_config):
            mock_config.return_value = {'test': 'value'}
            # Your test code here
            pass
        
        test_method()

    def test_patch_as_context_manager(self):
        """Using patch as a context manager"""
        with patch('base_torch.HGQ') as mock_hgq:
            mock_hgq.from_config.return_value = Mock()
            # Your test code here
            pass

    def test_patch_object_method(self):
        """Patching specific methods of objects"""
        layer = Mock()
        
        with patch.object(layer, 'build') as mock_build:
            layer.build((32, 10))
            mock_build.assert_called_once_with((32, 10))

    def test_property_mocking(self):
        """Mocking properties"""
        layer = Mock()
        
        # Mock a property
        type(layer).kernel_bw = PropertyMock(return_value=torch.tensor([4.0]))
        
        result = layer.kernel_bw
        self.assertEqual(result, torch.tensor([4.0]))

    def test_side_effects(self):
        """Using side effects for dynamic behavior"""
        mock_func = Mock()
        mock_func.side_effect = [1, 2, 3]  # Returns different values on successive calls
        
        self.assertEqual(mock_func(), 1)
        self.assertEqual(mock_func(), 2)
        self.assertEqual(mock_func(), 3)

    def test_exception_mocking(self):
        """Mocking exceptions"""
        mock_func = Mock()
        mock_func.side_effect = ValueError("Test error")
        
        with self.assertRaises(ValueError):
            mock_func()


if __name__ == '__main__':
    unittest.main()
    print("ALL GOOD!")