from utils import census_transform, hamming_distance, CostTensorCoordinates
import torch

class fSGM:
    def __init__(self,args):
        self.census_window = args.census_window
        self.path_directions = args.path_directions
        self.P1 = args.P1
        self.P2 = args.P2

    def compute_flow(self,frame_0,frame_1):
        #  'It first computes pixel-wise matching costs of corresponding pixels in two 
        #  frames for all disparities in the search space. '
        self.compute_matching_costs(frame_0,frame_1)

        return None

    def compute_matching_costs(self,frame_0,frame_1):
        census_0 = census_transform(frame_0)
        census_1 = census_transform(frame_1)
        height,width =census_0.size()
        self.displacement_costs = torch.zeros(height,width,self.census_window,self.census_window)
        coordinate_helper = CostTensorCoordinates(self.census_window,height,width)
        for i in range(height):
            for j in range(width):
                for k in range(self.census_window):
                    for l in range(self.census_window):
                        x,y = coordinate_helper.get_matching_frame_coordinate(i,j,k,l)
                        self.displacement_costs[i,j,k,l] = hamming_distance(census_0[i,j],census_1[x,y])
        print(self.displacement_costs)
