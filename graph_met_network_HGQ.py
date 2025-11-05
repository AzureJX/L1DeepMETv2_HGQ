import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from hgq.layers import QDense, QBatchNormalization, QUnaryFunctionLUT

from torch import nn

# from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv
from EdgeConv_HGQ import EdgeConv 

class GraphMETNetwork(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
    #def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
       
        self.datanorm = norm

        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        

        self.embed_continuous_dense = QDense(hidden_dim//2) # output
        # self.embed_continuous_dense.build((None, continuous_dim)) # input
        # kernal / input / bias config: kq_conf, iq_conf, bq_conf
        self.embed_continuous_elu = QUnaryFunctionLUT(activation='elu') # iq_conf, oq_conf

        self.embed_categorical_dense = QDense(hidden_dim//2)
        self.embed_categorical_elu = QUnaryFunctionLUT(activation='elu')

        self.encode_all_dense = QDense(hidden_dim)
        self.encode_all_elu = QUnaryFunctionLUT(activation='elu')
        self.bn_all = QBatchNormalization(axis=-1) # kq_conf, iq_conf, bq_conf
 
        self.conv_continuous = nn.ModuleList()        
        for _ in range(conv_depth):
            mesg = QDense(hidden_dim)
            conv_layer = EdgeConv(nn=mesg).jittable()
            bn_layer = QBatchNormalization(axis=-1)
            self.conv_continuous.append(nn.ModuleList([conv_layer, bn_layer]))

        self.output_dense1 = QDense(hidden_dim//2)
        self.output_elu = QUnaryFunctionLUT(activation='elu')
        self.output_dense2 = QDense(output_dim)

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x_cont, x_cat, edge_index, batch):
        # Normalize the input values within [0,1] range: pt, px, py, eta, phi, puppiWeight, pdgId, charge
        #norm = torch.tensor([1./2950., 1./2950, 1./2950, 1., 1., 1.]).to(device) 

        x_cont *= self.datanorm

        emb_cont = self.embed_continuous_dense(x_cont, training = True)
        emb_cont = self.embed_continuous_elu(emb_cont)

        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = torch.cat([emb_chrg, emb_pdg], dim=1)
        emb_cat = self.embed_categorical_dense(emb_cat, training = True)
        emb_cat = self.embed_categorical_elu(emb_cat)

        emb = torch.cat([emb_cat, emb_cont], dim=1)
        emb = self.encode_all_dense(emb, training = True)
        emb = self.encode_all_elu(emb)
        emb = self.bn_all(emb)

        # graph convolution for continuous variables
        # for co_conv in self.conv_continuous:
            # dynamic, evolving knn
            # emb = emb + co_conv[1](co_conv[0](emb, knn_graph(emb, k=20, batch=batch, loop=True)))
            # static
            # emb = emb + co_conv[1](co_conv[0](emb, edge_index))
        for conv_layer, bn_layer in self.conv_continuous:
            emb = emb + bn_layer(conv_layer(emb, edge_index))
                
        # out = self.output(emb)
        out = self.output_dense1(emb, training = True)
        out = self.output_elu(out)
        out = self.output_dense2(out, training = True)
        
        return out.squeeze(-1)
