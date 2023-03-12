import os.path
import torch
from torchvision import transforms
from PIL import Image


def gettorchtensor(path_name):
    _ , extension =  os.path.splitext(path_name)
    if extension == '.png':
        img = Image.open(path_name)
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img)
        

class Dataloader:
    def __init__(self,args):
        if args.dataset == 'Middlebury':
            self.frame_0_path = os.path.join('Middlebury/frames',args.scene,'frame10.png')
            self.frame_1_path = os.path.join('Middlebury/frames',args.scene,'frame11.png')
            self.gt_flow_path = os.path.join('Middlebury/gt-flow',args.scene,'flow10.flo')

    def getframes(self):
        frame_0 = gettorchtensor(self.frame_0_path)[:,:15,:12]
        frame_1 = gettorchtensor(self.frame_1_path)[:,:15,:12]
        return frame_0,frame_1