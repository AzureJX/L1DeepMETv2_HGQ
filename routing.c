typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix; // one 1D array

void node_to_edge(Matrix x, Matrix edge_index, float *x_i, float *x_j) {
    //assert \length(x_i) == E * F
    int N = x.rows;
    int F = x.cols; 
    int E = edge_index.cols;
    // Python: x_i = x[src] # [E, F]
    for (int e = 0; e < E; e++) {
        int src_node = edge_index.data[e];
        for (int f = 0; f < F; f++) {
            x_i[e * F + f] = x.data[src_node * F + f];
        }
    }
    // Python: x_j = x[tar] # [E, F]  
    for (int e = 0; e < E; e++) {
        int tar_node = edge_index.data[E+e];
        for (int f = 0; f < F; f++) {
            x_j[e * F + f] = x.data[tar_node * F + f];
        }
    }
} // output x_i, x_j

void max_aggregation(Matrix edge_feature, Matrix edge_index, Matrix output) {
    int N = output.rows;
    int out_channel = output.cols;
    int E = edge_index.cols;

    // output initialization?

    for (int e = 0; e < E; e++) {
        int node_idx = edge_index.data[e]; // src node
        
        for (int o = 0; o < out_channel; o++) {
            float new_val = edge_feature.data[e * out_channel + o];
            int old_value_index = node_idx * out_channel + o;
            if (new_val > output.data[old_value_index]) {
                output.data[old_value_index] = new_val; // update to bigger
            }
        }
    }
}