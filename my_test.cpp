#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdint>

// Forward declarations of the top functions from my.cpp
void node_to_edge_top(const float x[12], 
                      const uint32_t edge_index[10],
                      float x_i[15], 
                      float x_j[15]);

void max_aggregation_top(const float edge_feature[15],
                        const uint32_t edge_index[5],
                        float output[12]);

// Test function for node_to_edge
bool test_node_to_edge() {
    std::cout << "Testing node_to_edge..." << std::endl;
    
    // Test data: 4 nodes, 3 features each (total 12 elements)
    float x[12] = {
        1.0f, 2.0f, 3.0f,    // Node 0
        4.0f, 5.0f, 6.0f,    // Node 1
        7.0f, 8.0f, 9.0f,    // Node 2
        10.0f, 11.0f, 12.0f  // Node 3
    };
    
    // Edge index: [src_nodes, tar_nodes] (total 10 elements)
    // Edges: 0->1, 1->2, 2->0, 0->3, 3->1
    uint32_t edge_index[10] = {
        0, 1, 2, 0, 3,  // Source nodes
        1, 2, 0, 3, 1   // Target nodes
    };
    
    float x_i[15]; // 5 edges * 3 features
    float x_j[15]; // 5 edges * 3 features
    
    // Call the top function
    node_to_edge_top(x, edge_index, x_i, x_j);
    
    // Verify results
    std::cout << "Source node features (x_i):" << std::endl;
    for (int e = 0; e < 5; e++) {
        std::cout << "Edge " << e << ": ";
        for (int f = 0; f < 3; f++) {
            std::cout << x_i[e * 3 + f] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Target node features (x_j):" << std::endl;
    for (int e = 0; e < 5; e++) {
        std::cout << "Edge " << e << ": ";
        for (int f = 0; f < 3; f++) {
            std::cout << x_j[e * 3 + f] << " ";
        }
        std::cout << std::endl;
    }
    
    // Simple verification for edge 0 (0->1)
    bool passed = true;
    // Edge 0: src=0, tar=1
    for (int f = 0; f < 3; f++) {
        if (std::abs(x_i[f] - x[f]) > 1e-6) passed = false;         // Verify node 0 features
        if (std::abs(x_j[f] - x[3 + f]) > 1e-6) passed = false;     // Verify node 1 features
    }
    
    return passed;
}

// Test function for max_aggregation
bool test_max_aggregation() {
    std::cout << "\nTesting max_aggregation..." << std::endl;
    
    // Edge features: 5 edges, 3 channels each (total 15 elements)
    float edge_features[15] = {
        1.0f, 2.0f, 3.0f,   // Edge 0 -> Node 0
        4.0f, 1.0f, 2.0f,   // Edge 1 -> Node 1
        2.0f, 5.0f, 1.0f,   // Edge 2 -> Node 2
        3.0f, 3.0f, 4.0f,   // Edge 3 -> Node 0
        5.0f, 2.0f, 3.0f    // Edge 4 -> Node 3
    };
    
    // Source nodes for aggregation
    uint32_t edge_index[5] = {0, 1, 2, 0, 3};
    
    float output[12]; // 4 nodes * 3 channels
    
    // Call the top function
    max_aggregation_top(edge_features, edge_index, output);
    
    // Print results
    std::cout << "Aggregated node features:" << std::endl;
    for (int n = 0; n < 4; n++) {
        std::cout << "Node " << n << ": ";
        for (int c = 0; c < 3; c++) {
            std::cout << output[n * 3 + c] << " ";
        }
        std::cout << std::endl;
    }
    
    // Verify: Node 0 should have max of edge 0 and edge 3
    // Edge 0: [1, 2, 3], Edge 3: [3, 3, 4]
    // Expected: max(1,3), max(2,3), max(3,4) = 3, 3, 4
    bool passed = true;
    if (std::abs(output[0] - 3.0f) > 1e-6) passed = false;
    if (std::abs(output[1] - 3.0f) > 1e-6) passed = false;
    if (std::abs(output[2] - 4.0f) > 1e-6) passed = false;
    
    return passed;
}

int main() {
    std::cout << "=== Vitis HLS Graph Functions Testbench ===" << std::endl;
    
    bool test1_passed = test_node_to_edge();
    bool test2_passed = test_max_aggregation();
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "node_to_edge test: " << (test1_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "max_aggregation test: " << (test2_passed ? "PASSED" : "FAILED") << std::endl;
    
    if (test1_passed && test2_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}