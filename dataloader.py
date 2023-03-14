import os.path
from utils.file_io import get_torch_tensor

class Dataloader:
    def __init__(self,args):
        if args.dataset == 'Middlebury':
            self.frame_0_path = os.path.join('Middlebury/frames',args.scene,'frame10.png')
            self.frame_1_path = os.path.join('Middlebury/frames',args.scene,'frame11.png')
            self.gt_flow_path = os.path.join('Middlebury/gt-flow',args.scene,'flow10.flo')
            self.tl = (63,301)
            self.br = (101,333)
            #self.tl = (330,60)
            #self.br = (356,81)


    def get_frames(self):
        frame_0 = get_torch_tensor(self.frame_0_path)[:,self.tl[1]:self.br[1],self.tl[0]:self.br[0]]
        frame_1 = get_torch_tensor(self.frame_1_path)[:,self.tl[1]:self.br[1],self.tl[0]:self.br[0]]
        #frame_0 = get_torch_tensor(self.frame_0_path)
        #frame_1 = get_torch_tensor(self.frame_1_path)
        return frame_0,frame_1
    def get_paths(self):
        return (self.frame_0_path,self.frame_1_path,self.gt_flow_path)
