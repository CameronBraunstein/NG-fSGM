import os.path
import torch
from torchvision import transforms
from PIL import Image

def save_flow_file(args,flow_prediction):
    return "TO CONTINUE: FLOW FILE FORMATTING"

def save_cost_tensor(args,predictor):
    path_name = '{}_cost_tensor.pth'.format(args.scene)
    cost_tensor = getattr(predictor,'total_flow_costs')
    torch.save(cost_tensor,path_name)
    return path_name

def load_tensor(path_name):
    return torch.load(path_name)

def get_torch_tensor(path_name):
    _ , extension =  os.path.splitext(path_name)
    if extension == '.png':
        img = Image.open(path_name)
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img)
