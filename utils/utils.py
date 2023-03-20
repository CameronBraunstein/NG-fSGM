
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from multiprocessing import Pool


# Based off of extremely helpful stack overflow post:
# https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def census_transform(img, img_padding=1):
    transform = transforms.Grayscale()
    img = transform(img)
    transform = transforms.Pad(1,padding_mode='symmetric')
    for i in range(img_padding):
        img = transform(img)
    img = np.asarray(img)
    height,width = img.shape
    census = np.zeros((height-2, width-2),dtype=np.uint8)
    center_pixels = img[1:height-1, 1:width-1]
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]
    for u,v in offsets:
        census = (census << 1) | (img[v:v+height-2, u:u+width-2] >= center_pixels)
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
        self.calculate_parallel()
    def get_data_costs(self,i,j):
        #Calculate displacement costs (C(p,o) in Zhang et al.)
        data_costs = np.zeros((self.displacement_window,self.displacement_window))
        base_census = self.census_0[i,j]
        for k in range(self.displacement_window):
            for l in range(self.displacement_window):
                data_costs[k,l] = hamming_distance(base_census,self.census_1[i+k,j+l])
        return data_costs
    def calculate_parallel(self):
        partitions = os.cpu_count()-2
        p = Pool(partitions)
        max_height,max_width = self.census_0.shape

        #bases = [self.census_0[i*(max_height//partitions):min((i+1)*(max_height//partitions),max_height),:] for i in range(partitions)]
        bases = [self.census_0 for i in range(partitions)]
        matches = [self.census_1 for i in range(partitions)]
        displacement_windows = [self.displacement_window for i in range(partitions)]
        min_heights = [i*(max_height//partitions + 1 ) for i in range(partitions)]
        max_heights = [min((i+1)*(max_height//partitions + 1 ),max_height) for i in range(partitions)]
        min_widths = [0 for i in range(partitions)]
        max_widths = [max_width for i in range(partitions)]

        cost_chunks = p.starmap(data_costs_in_range, zip(bases,matches,displacement_windows,min_heights,max_heights,min_widths,max_widths))
        self.data_costs = np.zeros((max_height,max_width,self.displacement_window,self.displacement_window))
        for costs,coords in cost_chunks:
            self.data_costs[coords[0]:coords[1], coords[2]:coords[3],:,:] = costs
    

    

def data_costs_in_range(base_census,match_census,displacement_window,height_min,height_max,width_min,width_max):
    data_costs = np.zeros((height_max-height_min,width_max-width_min,displacement_window,displacement_window))
    for i in range(height_min,height_max):
        for j in range(width_min,width_max):
            base = base_census[i,j]
            for k in range(displacement_window):
                for l in range(displacement_window):
                    data_costs[i-height_min,j-width_min,k,l] = hamming_distance(base,match_census[i+k,j+l])
    return data_costs,(height_min,height_max,width_min,width_max)





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
    width,height = directional_penalties.shape
    directional_regularizers = np.full((width,height),float('inf'))
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
    directional_regularizers = directional_regularizers-global_min
    return directional_regularizers
    
def get_offset_from_window_size(window_size):
    return window_size //2


def view_torch_tensor(img):
    sample = transforms.ToPILImage()(img)
    sample.show()