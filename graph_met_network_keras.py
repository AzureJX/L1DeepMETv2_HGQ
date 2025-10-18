import tensorflow as tf
import keras
from keras import layers, Model
import numpy as np

# NOTE: These torch-geometric and torch-cluster imports cannot be directly converted
# You'll need to find TensorFlow/Keras equivalents or implement custom layers
# CANNOT CONVERT: from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv
# CANNOT CONVERT: from torch_cluster import radius_graph, knn_graph

class GraphMETNetwork(Model):
    def __init__(self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
        
        self.datanorm = norm
        
        # Embedding layers with Keras
        self.embed_charge = layers.Embedding(3, hidden_dim//4)
        self.embed_pdgid = layers.Embedding(7, hidden_dim//4)
        
        # Keras sequential layers
        self.embed_continuous = keras.Sequential([
            layers.Dense(hidden_dim//2),
            layers.ELU(),
            # layers.BatchNormalization()  # uncomment if it starts overtraining
        ])
        self.embed_categorical = keras.Sequential([
            layers.Dense(hidden_dim//2),
            layers.ELU(),
            # layers.BatchNormalization()
        ])
        self.encode_all = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.ELU()
        ])
        self.bn_all = layers.BatchNormalization()
        
        self.conv_continuous = []
        for i in range(conv_depth):
            # CANNOT CONVERT: EdgeConv(nn=mesg).jittable()
            # This would need a custom implementation or TF Graph Nets
            mesg = keras.Sequential([layers.Dense(hidden_dim)])
            conv_layer = {
                'message_net': mesg,
                'batch_norm': layers.BatchNormalization()
            }
            self.conv_continuous.append(conv_layer)
        
        # Keras sequential output layers
        self.output_layers = keras.Sequential([
            layers.Dense(hidden_dim//2),
            layers.ELU(),
            layers.Dense(output_dim)
        ])
        
        self.pdgs = [1, 2, 11, 13, 22, 130, 211]
    
    def call(self, inputs, training=None):
        """
        Expected input format: 
        inputs = {
            'x_cont': continuous features,
            'x_cat': categorical features,
            'edge_index': edge connectivity (CANNOT BE PROCESSED without custom EdgeConv),
            'batch': batch indices (CANNOT BE PROCESSED without torch-geometric equivalent)
        }
        """
        x_cont = inputs['x_cont']
        x_cat = inputs['x_cat']
        edge_index = inputs['edge_index']
        batch = inputs['batch']
        
        x_cont = x_cont * self.datanorm
        
        emb_cont = self.embed_continuous(x_cont)
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)
        
        # Torch to tf
        pdg_remap = tf.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = tf.where(
                tf.equal(pdg_remap, pdgval),
                tf.fill(tf.shape(pdg_remap), i),
                pdg_remap
            )
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(tf.concat([emb_chrg, emb_pdg], axis=1))
        #emb_cat = self.embed_categorical(tf.concat([emb_chrg, emb_pdg, emb_pv], axis=1))
        emb = tf.concat([emb_cat, emb_cont], axis=1)
        emb = self.encode_all(emb)
        emb = self.bn_all(emb, training=training)
        
        # Edgeconv
        # Requires custom implementation of EdgeConv
        # or using TF Graph Neural Network libraries?
        """
        for co_conv in self.conv_continuous:
            # CANNOT CONVERT: EdgeConv operations
            # Original: emb = emb + co_conv[1](co_conv[0](emb, edge_index))
            # Would need custom EdgeConv layer implementation
            pass
        """        
        out = self.output_layers(emb)
        
        return tf.squeeze(out, axis=-1)