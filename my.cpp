#include <cstdint>

struct node_to_edge_config {
    static const unsigned N = 4;      // Number of nodes
    static const unsigned F = 3;      // Feature dimension
    static const unsigned E = 5;      // Number of edges
};

struct max_aggregation_config {
    static const unsigned N = 4;
    static const unsigned E = 5;
    static const unsigned out_channel = 3;
};

template <typename data_T, typename res_T, typename CONFIG_T>
void node_to_edge(const data_T x[CONFIG_T::N * CONFIG_T::F], 
                  const uint32_t edge_index[2 * CONFIG_T::E],
                  res_T x_i[CONFIG_T::E * CONFIG_T::F], 
                  res_T x_j[CONFIG_T::E * CONFIG_T::F]) {
    
    // Extract source node features
    for (int e = 0; e < CONFIG_T::E; e++) {
        #pragma HLS UNROLL
        uint32_t src_node = edge_index[e];
        for (int f = 0; f < CONFIG_T::F; f++) {
            #pragma HLS UNROLL
            x_i[e * CONFIG_T::F + f] = x[src_node * CONFIG_T::F + f];
        }
    }
    
    // Extract target node features
    for (int e = 0; e < CONFIG_T::E; e++) {
        #pragma HLS UNROLL
        uint32_t tar_node = edge_index[CONFIG_T::E + e];
        for (int f = 0; f < CONFIG_T::F; f++) {
            #pragma HLS UNROLL
            x_j[e * CONFIG_T::F + f] = x[tar_node * CONFIG_T::F + f];
        }
    }
}

template <typename data_T, typename res_T, typename CONFIG_T>
void max_aggregation(const data_T edge_feature[CONFIG_T::E * CONFIG_T::out_channel],
                     const uint32_t edge_index[CONFIG_T::E], // *2
                     res_T output[CONFIG_T::N * CONFIG_T::out_channel]) {
    
    for (int i = 0; i < CONFIG_T::N * CONFIG_T::out_channel; i++) {
        #pragma HLS UNROLL
        output[i] = static_cast<res_T>(-3.4e38f);
    }
    
    for (int e = 0; e < CONFIG_T::E; e++) {
        #pragma HLS UNROLL
        uint32_t node_idx = edge_index[e];
        
        for (int o = 0; o < CONFIG_T::out_channel; o++) {
            #pragma HLS UNROLL
            data_T new_val = edge_feature[e * CONFIG_T::out_channel + o];
            int old_value_index = node_idx * CONFIG_T::out_channel + o;
            
            if (new_val > output[old_value_index]) {
                output[old_value_index] = static_cast<res_T>(new_val);
            }
        }
    }
}

// Wrapper functions for HLS top functions
void node_to_edge_top(const float x[12], 
                      const uint32_t edge_index[10],
                      float x_i[15], 
                      float x_j[15]) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=edge_index offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=x_i offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=x_j offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=return
    
    node_to_edge<float, float, node_to_edge_config>(x, edge_index, x_i, x_j);
}

void max_aggregation_top(const float edge_feature[15],
                        const uint32_t edge_index[5],
                        float output[12]) {
    #pragma HLS INTERFACE m_axi port=edge_feature offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=edge_index offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return
    
    max_aggregation<float, float, max_aggregation_config>(edge_feature, edge_index, output);
}