import os.path
from utils.file_io import get_as_PIL_image

class Dataloader:
    def __init__(self,args):
        if args.dataset == 'Middlebury':
            self.frame_0_path = os.path.join('Middlebury/frames',args.scene,'frame10.png')
            self.frame_1_path = os.path.join('Middlebury/frames',args.scene,'frame11.png')
            self.gt_flow_path = os.path.join('Middlebury/gt-flow',args.scene,'flow10.flo')
            self.tl = (55,282)
            self.br = (121,353)
            #self.tl = (330,60)
            #self.br = (356,81)


    def get_frames(self):
        frame_0 = get_as_PIL_image(self.frame_0_path)
        frame_1 = get_as_PIL_image(self.frame_1_path)
        if True:
            frame_0 = frame_0.crop((self.tl[0],self.tl[1],self.br[0],self.br[1]))
            frame_1 = frame_1.crop((self.tl[0],self.tl[1],self.br[0],self.br[1]))
        return frame_0,frame_1
    def get_paths(self):
        return (self.frame_0_path,self.frame_1_path,self.gt_flow_path)
