import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from model import Gmodel
from torch.utils.mobile_optimizer import optimize_for_mobile

model = Gmodel().to('cpu')
model_path = 'model_checkpoints/fusion_10'
ptl_model_path = 'model_checkpoints/gmodel_timm_effnet4.ptl'
model.load_state_dict(torch.load(model_path))
model.eval()
example = torch.rand(1,3,256,256)
traced_script_module = torch.jit.trace(model, example)
optimize_model = optimize_for_mobile(traced_script_module)
optimize_model._save_for_lite_interpreter(ptl_model_path)