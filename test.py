import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from typing import Union, Tuple

# Import our custom EdgeConv implementation
# (Assuming the EdgeConv class from the previous artifact is available)

class EdgeConv(nn.Module):
    """
    Args:
        in_channels (int): Number of input features per node
        out_channels (int): Number of output features per node
        aggr (str): 'max', 'mean', 'sum', 'add'
    """
    
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(EdgeConv, self).__init__()
        self.aggr = aggr
        
        # Dense MLP with depth=1
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x, edge_index, batch=None):
        """        
        Args:
            x: [N, F] - node features for all graphs, F = in_channels
            edge_index: [2, E] - edge indices for all graphs
            batch: [N] - batch assignment for each node (optional)
            
        Returns:
            out: [N, out_channels] - output node features
        """
        source, target = edge_index
        
        # Collect node features for source and target
        x_i = x[source] # [E, F]
        # x_i = torch.gather(x, 0, row.unsqueeze(1).expand(-1, x.size(1)))
        x_j = x[target] # [E, F]
        # x_j = torch.gather(x, 0, col.unsqueeze(1).expand(-1, x.size(1)))
        
        edge_feat = torch.cat([x_i, x_j - x_i], dim=1)  # [E, 2F]
        
        edge_feat = self.mlp(edge_feat)  # shape becomes [E, out_channels]
        
        N = x.size(0)
        out = torch.zeros(N, edge_feat.size(1), device=x.device, dtype=x.dtype)
        # out: shape [N, out]
        # Expand row indices for scatter operations
        source_expanded = source.unsqueeze(1).expand(-1, edge_feat.size(1)) # shape [E, out]
        # Aggregate over outcoming edges for each node
        if self.aggr == 'max':
            out.scatter_reduce_(0, source_expanded, edge_feat, reduce='amax', include_self=False)
        elif self.aggr == 'mean':
            out.scatter_reduce_(0, source_expanded, edge_feat, reduce='mean', include_self=False)
        elif self.aggr in ['sum', 'add']:
            out.scatter_add_(0, source_expanded, edge_feat)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggr}")
        
        return out


def create_test_data(num_nodes=100, in_channels=3, seed=42):
    """Create reproducible test data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create node features
    x = torch.randn(num_nodes, in_channels)
    
    # Create a simple ring graph with some random edges
    edges = []
    
    # Ring edges (each node connects to next)
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        edges.append([i, next_node])
        edges.append([next_node, i])  # Make undirected
    
    # Add some random edges
    num_random_edges = num_nodes // 2
    for _ in range(num_random_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:
            edges.append([src, dst])
            edges.append([dst, src])  # Make undirected
    
    # Remove duplicates and convert to tensor
    edges = list(set(tuple(edge) for edge in edges))
    edge_index = torch.tensor(edges).t()
    
    return x, edge_index


def test_custom_vs_pyg():
    """Test custom EdgeConv against PyTorch Geometric's implementation."""
    
    print("=" * 60)
    print("TESTING CUSTOM EDGECONV vs PYTORCH GEOMETRIC EDGECONV")
    print("=" * 60)
    
    # Test parameters
    num_nodes = 50
    in_channels = 4
    out_channels = 16
    
    # Create test data
    x, edge_index = create_test_data(num_nodes, in_channels)
    
    print(f"Test setup:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Node features shape: {x.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    
    # Test different aggregation methods
    for aggr in ['max', 'mean', 'add']:
        print(f"\n{'-' * 40}")
        print(f"Testing aggregation: {aggr}")
        print(f"{'-' * 40}")
        
        # Create identical MLPs for fair comparison
        mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        
        # Custom EdgeConv
        custom_conv = EdgeConv(in_channels, out_channels, aggr=aggr)
        # Copy the MLP weights to ensure identical initialization
        custom_conv.mlp = mlp
        
        # Try to import PyTorch Geometric
        try:
            from torch_geometric.nn import EdgeConv as PyGEdgeConv
            
            # PyTorch Geometric EdgeConv (create identical MLP)
            mlp_copy = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            )
            # Copy weights from the original MLP
            mlp_copy.load_state_dict(mlp.state_dict())
            
            pyg_conv = PyGEdgeConv(mlp_copy, aggr=aggr)
            
            # Forward pass
            with torch.no_grad():
                custom_output = custom_conv(x, edge_index)
                pyg_output = pyg_conv(x, edge_index)
            
            # Compare outputs
            diff = torch.abs(custom_output - pyg_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"Custom output shape: {custom_output.shape}")
            print(f"PyG output shape: {pyg_output.shape}")
            print(f"Max absolute difference: {max_diff:.8f}")
            print(f"Mean absolute difference: {mean_diff:.8f}")
            
            if max_diff < 1e-6:
                print("✅ PASS: Outputs match within tolerance!")
            else:
                print("❌ FAIL: Outputs differ significantly!")
                print(f"Custom output sample: {custom_output[:3, :3]}")
                print(f"PyG output sample: {pyg_output[:3, :3]}")
            
        except ImportError:
            print("⚠️  PyTorch Geometric not available, testing custom implementation only...")
            
            # Test custom implementation
            with torch.no_grad():
                custom_output = custom_conv(x, edge_index)
            
            print(f"Custom output shape: {custom_output.shape}")
            print(f"Output range: [{custom_output.min():.4f}, {custom_output.max():.4f}]")
            print(f"Output mean: {custom_output.mean():.4f}")
            print(f"Output std: {custom_output.std():.4f}")
            
            # Basic sanity checks
            assert custom_output.shape == (num_nodes, out_channels), f"Wrong output shape!"
            assert not torch.isnan(custom_output).any(), "Output contains NaN!"
            assert not torch.isinf(custom_output).any(), "Output contains Inf!"
            print("✅ Basic sanity checks passed!")


def test_gradient_flow():
    """Test that gradients flow correctly through the custom implementation."""
    print(f"\n{'=' * 40}")
    print("TESTING GRADIENT FLOW")
    print(f"{'=' * 40}")
    
    num_nodes = 20
    in_channels = 3
    out_channels = 8
    
    x, edge_index = create_test_data(num_nodes, in_channels)
    x.requires_grad_(True)
    
    # Create model
    model = EdgeConv(in_channels, out_channels, aggr='max')
    
    # Forward pass
    output = model(x, edge_index)
    
    # Compute a simple loss
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradients - Mean: {x.grad.mean():.6f}, Std: {x.grad.std():.6f}")
    
    # Check model parameter gradients
    param_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grads.append(param.grad.abs().mean().item())
            print(f"{name} gradient mean: {param.grad.abs().mean():.6f}")
        else:
            print(f"{name} has no gradient!")
    
    if all(grad > 0 for grad in param_grads):
        print("✅ All parameters have non-zero gradients!")
    else:
        print("❌ Some parameters have zero gradients!")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print(f"\n{'=' * 40}")
    print("TESTING EDGE CASES")
    print(f"{'=' * 40}")
    
    # Test 1: Single node
    print("Test 1: Single node with self-loop")
    x_single = torch.randn(1, 3)
    edge_single = torch.tensor([[0], [0]])  # Self-loop
    model = EdgeConv(3, 8, aggr='max')
    
    try:
        output_single = model(x_single, edge_single)
        print(f"✅ Single node test passed. Output shape: {output_single.shape}")
    except Exception as e:
        print(f"❌ Single node test failed: {e}")
    
    # Test 2: Empty graph (no edges)
    print("\nTest 2: Graph with no edges")
    x_empty = torch.randn(5, 3)
    edge_empty = torch.empty((2, 0), dtype=torch.long)
    
    try:
        output_empty = model(x_empty, edge_empty)
        print(f"✅ Empty edges test passed. Output shape: {output_empty.shape}")
        print(f"Output (should be zeros): {output_empty.abs().max().item():.6f}")
    except Exception as e:
        print(f"❌ Empty edges test failed: {e}")
    
    # Test 3: Invalid aggregation
    print("\nTest 3: Invalid aggregation method")
    try:
        invalid_model = EdgeConv(3, 8, aggr='invalid')
        x_test = torch.randn(10, 3)
        edge_test = torch.tensor([[0, 1], [1, 2]])
        output = invalid_model(x_test, edge_test)
        print("❌ Should have failed with invalid aggregation!")
    except ValueError as e:
        print(f"✅ Correctly caught invalid aggregation: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def benchmark_performance():
    """Basic performance benchmark."""
    print(f"\n{'=' * 40}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'=' * 40}")
    
    import time
    
    # Test with different graph sizes
    sizes = [100, 500, 1000, 2000]
    
    for num_nodes in sizes:
        x, edge_index = create_test_data(num_nodes, in_channels=16)
        model = EdgeConv(16, 32, aggr='max')
        
        # Warmup
        for _ in range(5):
            _ = model(x, edge_index)
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            output = model(x, edge_index)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        print(f"Nodes: {num_nodes:4d}, Edges: {edge_index.size(1):5d}, "
              f"Time: {avg_time:.2f}ms, "
              f"Output shape: {output.shape}")


if __name__ == "__main__":
    # Run all tests
    test_custom_vs_pyg()
    test_gradient_flow()
    test_edge_cases()
    benchmark_performance()
    
    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETED!")
    print(f"{'=' * 60}")