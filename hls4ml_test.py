import torch
import torch.nn as nn
# import hls4ml
import numpy as np
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.fc(x)

# my_model = SimpleModel()
# dummy_input = torch.randn(1, 2)

# hls_model = hls4ml.converters.convert_from_pytorch_model(
#     model=my_model, 
#     input_shape=(2,),
#     output_dir='hls_output'
# )

# hls_model.compile()

model = SimpleModel()
model.eval()

X_input = np.random.rand(1,2)

pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

config = config_from_pytorch_model(model, (2,))
output_dir = './hls_output/hls4mlprj_pytorch_api_linear_Vitis_io_parallel'

hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend='Vitis', io_type='io_parallel')
hls_model.write()
