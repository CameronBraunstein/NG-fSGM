
import torch
import torchvision.transforms as transforms
from PIL import Image


# Based off of extremely helpful stack overflow post:
# https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def census_transform(img):
    transform = transforms.Grayscale()
    img = transform(img)
    transform = transforms.Pad(1,padding_mode='symmetric')
    img = transform(img)
    img_size = img.size()
    height,width = img_size[1],img_size[2]
    census = torch.zeros(size=(height-2, width-2),dtype=torch.uint8)
    center_pixels = img[0,1:height-1, 1:width-1]
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]
    for u,v in offsets:
        census = (census << 1) | (img[0,v:v+height-2, u:u+width-2] >= center_pixels)
    return census

def hamming_distance(a,b):
    return bin(a ^ b).count("1")


class CostTensorCoordinates:
    def __init__(self,census_window,max_height,max_width):
        if census_window%2 == 0:
            raise Exception("Census window must be an odd number")
        self.window_offset = census_window//2
        self.max_height = max_height
        self.max_width = max_width
    def get_matching_frame_coordinate(self, frame_x,frame_y, window_x,window_y):
        shifted = False
        if frame_x-self.window_offset<0:
            x = window_x
            shifted = True
        elif frame_x + self.window_offset >= self.max_height:
            x = self.max_height - (self.window_offset*2 +1) + window_x
            shifted = True
        else:
            x = frame_x - self.window_offset + window_x 
        
        if frame_y-self.window_offset<0:
            y = window_y
            shifted = True
        elif frame_y + self.window_offset >= self.max_width:
            shifted = True
            y = self.max_width - (self.window_offset*2 +1) + window_y
        else:
            y = frame_y - self.window_offset + window_y

        if shifted:
            print(frame_x,frame_y, window_x,window_y)
            print(x,y)
            print('corrd shifts')

        return (x,y)



def view_torch_tensor(img):
    sample = transforms.ToPILImage()(img)
    sample.show()