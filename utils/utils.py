
import torch
import torchvision.transforms as transforms
from PIL import Image


# Based off of extremely helpful stack overflow post:
# https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def census_transform(img, img_padding=1):
    transform = transforms.Grayscale()
    img = transform(img)
    transform = transforms.Pad(1,padding_mode='symmetric')
    for i in range(img_padding):
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


#In the future, expand to subclasses with 
#various data costs Mutual information, etc.,
#For now, we exclusively use the census
class DataCostCalculator:
    def __init__(self,frame_0,frame_1,displacement_window):
        self.displacement_window = displacement_window
        self.census_0 = census_transform(frame_0)
        self.census_1 = census_transform(frame_1,img_padding=get_offset_from_window_size(self.displacement_window)+1)
    def get_data_costs(self,i,j):
        #Calculate displacement costs (C(p,o) in Zhang et at.)
        data_costs = torch.zeros(self.displacement_window,self.displacement_window)
        base_census = self.census_0[i,j]
        for k in range(self.displacement_window):
            for l in range(self.displacement_window):
                data_costs[k,l] = hamming_distance(base_census,self.census_1[i+k,j+l])
        return data_costs




class DirectionalRegularizerCache:
    def __init__(self):
        self.regularization_penalties_dict = {}
    def format_key(self,point,direction):
        return ','.join(tuple(map(str,point)))+','+(','.join(tuple(map(str,direction))))
    
    def add_regularization_penalty(self,point,direction,penalties):
        key = self.format_key(point,direction)
        self.regularization_penalties_dict[key] = penalties
    
    def get_regularization_penalty(self,point,direction):
        key = self.format_key(point,direction)
        if key in self.regularization_penalties_dict:
            return self.regularization_penalties_dict[key]
        else:
            return None
        
    def clear_keys_if_last_use(self,point,direction_vectors):
        for (u,v) in direction_vectors:
            key = self.format_key((point[0]-u,point[1]-v),(u,v))
            if key in self.regularization_penalties_dict:
                del self.regularization_penalties_dict[key]


def calculate_Potts_directional_regularizers(directional_penalties,P1,P2):
    width,height = directional_penalties.size()
    directional_regularizers = torch.full((width,height),float('inf'))
    global_min = float('inf')
    for i in range(width):
        for j in range(height):
            if directional_penalties[i,j]<global_min:
                global_min = directional_penalties[i,j]
            for k in range(width):
                for l in range(height):
                    squared_dist = (i-k)**2 + (j-l)**2
                    if squared_dist == 0:
                        if directional_penalties[i,j] < directional_regularizers[k,l]:
                            directional_regularizers[k,l] = directional_penalties[i,j]
                    elif squared_dist < 2:
                        if directional_penalties[i,j]+ P1 < directional_regularizers[k,l]:
                            directional_regularizers[k,l] = directional_penalties[i,j] + P1
                    else:
                        if directional_penalties[i,j]+ P2 < directional_regularizers[k,l]:
                            directional_regularizers[k,l] = directional_penalties[i,j] + P2
    directional_regularizers = torch.sub(directional_regularizers,global_min)
    return directional_regularizers


             



    
def get_offset_from_window_size(window_size):
    return window_size //2


def view_torch_tensor(img):
    sample = transforms.ToPILImage()(img)
    sample.show()

class CostTensorCoordinates:
    def __init__(self,census_window,max_height,max_width):
        self.window_offset = census_window//2
        self.max_height = max_height
        self.max_width = max_width
    def get_matching_frame_coordinates(self, frame_x,frame_y, window_x,window_y):
        if frame_x-self.window_offset<0:
            x = window_x
        elif frame_x + self.window_offset >= self.max_height:
            x = self.max_height - (self.window_offset*2 +1) + window_x
        else:
            x = frame_x - self.window_offset + window_x 
        
        if frame_y-self.window_offset<0:
            y = window_y
        elif frame_y + self.window_offset >= self.max_width:
            y = self.max_width - (self.window_offset*2 +1) + window_y
        else:
            y = frame_y - self.window_offset + window_y
        return (x,y)