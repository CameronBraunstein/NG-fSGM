import torchvision.transforms as transforms
from utils.utils import get_offset_from_window_size
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
class DataCosts:
    def __init__(self,frame_0,frame_1,displacement_window):
        self.displacement_window = displacement_window
        self.census_0 = census_transform(frame_0)
        self.census_1 = census_transform(frame_1,img_padding=get_offset_from_window_size(self.displacement_window)+1)
        partitions = os.cpu_count()-2
        
        max_height,max_width = self.census_0.shape

        #bases = [self.census_0[i*(max_height//partitions):min((i+1)*(max_height//partitions),max_height),:] for i in range(partitions)]
        bases = [self.census_0 for i in range(partitions)]
        matches = [self.census_1 for i in range(partitions)]
        displacement_windows = [self.displacement_window for i in range(partitions)]
        min_heights = [index[0] for index in np.array_split(range(max_height), partitions)]
        max_heights = [index[-1]+1 for index in np.array_split(range(max_height), partitions)]
        min_widths = [0 for i in range(partitions)]
        max_widths = [max_width for i in range(partitions)]

        p = Pool(partitions)
        cost_chunks = p.starmap(data_costs_in_range, zip(bases,matches,displacement_windows,min_heights,max_heights,min_widths,max_widths))
        self.data_costs = np.zeros((max_height,max_width,self.displacement_window,self.displacement_window))
        for costs,coords in cost_chunks:
            self.data_costs[coords[0]:coords[1], coords[2]:coords[3],:,:] = costs
    def get_data_costs(self,i,j):
        return self.data_costs[i,j,:,:]
        
    

def data_costs_in_range(base_census,match_census,displacement_window,height_min,height_max,width_min,width_max):
    data_costs = np.zeros((height_max-height_min,width_max-width_min,displacement_window,displacement_window))
    for i in range(height_min,height_max):
        for j in range(width_min,width_max):
            base = base_census[i,j]
            for k in range(displacement_window):
                for l in range(displacement_window):
                    #Calculate displacement costs (C(p,o) in Zhang et al.)
                    data_costs[i-height_min,j-width_min,k,l] = hamming_distance(base,match_census[i+k,j+l])
    return data_costs,(height_min,height_max,width_min,width_max)
