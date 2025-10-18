void matrix_multiply(float* input, float* weight, float* output, 
                    int input_size, int output_size) {
    // input: [input_size], weight: [input_size][output_size], output: [output_size]
    for(int i = 0; i < output_size; i++) {
        output[i] = 0;
        for(int j = 0; j < input_size; j++) {
            output[i] += input[j] * weight[j * output_size + i];
        }
        // + bias
        output[i] += bias[i];
        // ReLU
        if(output[i] < 0) output[i] = 0;
    }
}

void aggregate(float* output, float* new_values, int node_idx, int channels) {
    for(int c = 0; c < channels; c++) {
        // max
        if(new_values[c] > output[node_idx * channels + c]) {
            output[node_idx * channels + c] = new_values[c];
        }
    }
}

void edgeconv_forward(
    float* x,           // [n1f1,n1f2...n1fF, n2f1,n2f2...]
    int* edge_index,    // [s1,s2...sE, t1,t2...tE]
    float* weights,     // MLP weight
    float* output,      // [N, out_channels]
    int N, int E, int F, int out_channels
) {
    for(int e = 0; e < E; e++) {
        int src = edge_index[e]; // [0], [1]
        int tar = edge_index[e + E]; // [E], [E+1]
        
        float edge_feat[2*F];
        for(int f = 0; f < F; f++) {
            edge_feat[f] = x[src*F + f];           // x_i
            edge_feat[f+F] = x[tar*F + f] - x[src*F + f]; // x_j - x_i
        }
        
        // 3. MLP前向传播（矩阵乘法）
        float mlp_out[out_channels];
        matrix_multiply(edge_feat, weights, mlp_out, 2*F, out_channels);
        
        // 4. 聚合到输出节点
        aggregate(output, mlp_out, src, out_channels);
    }
}