from utils.utils import DataCostCalculator, DirectionalRegularizerCache, calculate_Potts_directional_regularizers, get_offset_from_window_size

from utils.data_cost_calculator import DataCosts
from utils.regularization_model import RegularizationModel

import numpy as np
from multiprocessing import Pool

#Based on description in Hierarchical Scan-Line Dynamic Programming for 
# Optical Flow Using Semi-Global Matching, Simon Hermann & Reinhard Klette

class fSGM:
    def __init__(self,args):
        self.displacement_window = args.displacement_window
        self.path_directions = args.path_directions
        self.regularization_model = RegularizationModel(args)


    def compute_flow(self,frame_0,frame_1):
        

        data_costs_cache = DataCosts(frame_0,frame_1,self.displacement_window)

        process_count = self.path_directions #One process for each direction, this can be expanded if necessary

        costs = [data_costs_cache for i in range(process_count)]
        direction_vectors = [(1,0),(1,1),(0,1),(1,-1),(-1,0),(-1,-1),(0,-1),(-1,1)]
        is_forward_passes = [True,True,True,True,False,False,False,False]
        regularizers = [self.regularization_model for i in range(process_count)]

        width,height = frame_0.size
        heights= [height for i in range(process_count)]
        widths= [width for i in range(process_count)]
        displacement_windows = [self.displacement_window for i in range(process_count)]

        p = Pool(process_count)
        directional_penalties = p.starmap(compute_directional_penalty, zip(costs,direction_vectors,is_forward_passes,regularizers,heights,widths,displacement_windows))
        self.total_flow_costs = np.zeros((height,width,self.displacement_window,self.displacement_window))
        for penalty in directional_penalties:
            self.total_flow_costs = np.add(self.total_flow_costs,penalty)

        #THIS CODE WILL BE OBSOLETE AFTER PARALLELIZATION

        # data_cost_calc = DataCostCalculator(frame_0,frame_1,self.displacement_window)
        # #Forward pass
        # print('Beginning Forward Pass')
        # self.forward_or_backward_pass(height,width,data_cost_calc,is_forward=True)
        # #Backward pass
        # print('Beginning Backward Pass')
        # self.forward_or_backward_pass(height,width,data_cost_calc,is_forward=False)

        self.flow = np.zeros((height,width,2))
        self.get_flows_from_total_costs(height,width)

        return self.flow
    
    def get_flows_from_total_costs(self,height,width):
        window_offset = get_offset_from_window_size(self.displacement_window)
        for i in range(height):
            for j in range(width):
                argmin = np.argmin(self.total_flow_costs[i,j,:,:])
                self.flow[i,j,0] = (argmin % self.displacement_window) -window_offset
                self.flow[i,j,1] = (argmin // self.displacement_window) -window_offset

    
    def forward_or_backward_pass(self,height,width,data_cost_calc,is_forward):
        regularizer_cache = DirectionalRegularizerCache()
        #Calibrate for the forward or backward pass
        if is_forward:
            height_range = range(0,height)
            width_range = range(0,width)
            direction_vectors = [(1,0),(1,1),(0,1),(1,-1)]
        else:
            height_range = range(height-1, -1, -1)
            width_range = range(width-1,  -1, -1)
            direction_vectors = [(-1,0),(-1,-1),(0,-1),(-1,1)]
        for i in height_range:
            if i % 10 == 0:
                print('On row ',i)
            for j in width_range:
                data_costs =data_cost_calc.get_data_costs(i,j)
                for (u,v) in direction_vectors:
                    directional_regularizers = regularizer_cache.get_regularization_penalty((i-u,j-v),(u,v))
                    if directional_regularizers is not None:
                        directional_penalties = np.add(data_costs,directional_regularizers)
                    else:
                        directional_penalties = data_costs

                    #Update final costs
                    self.total_flow_costs[i,j,:,:] = np.add(self.total_flow_costs[i,j,:,:],directional_penalties)

                    #Compute regularization penalties 
                    if self.penalty_model == 'Potts':
                        new_directional_regularizers = calculate_Potts_directional_regularizers(directional_penalties,self.P1,self.P2)
                    else:
                        raise Exception("Only supporting Potts model for now")
                    
                    regularizer_cache.add_regularization_penalty((i,j),(u,v),new_directional_regularizers)
                
                regularizer_cache.clear_keys_if_last_use((i,j),direction_vectors)
        return


def compute_directional_penalty(data_costs_cache,direction_vector,is_forward_pass,regularizer,height,width,displacement_window):
    regularizer_cache = DirectionalRegularizerCache()
    directional_penalty = np.zeros((height,width,displacement_window,displacement_window))
    #Calibrate for the forward or backward pass
    if is_forward_pass:
        height_range = range(0,height)
        width_range = range(0,width)
    else:
        height_range = range(height-1, -1, -1)
        width_range = range(width-1,  -1, -1)

    u,v = direction_vector
    print('begin: ', u,v)
    for i in height_range:
        if i % 10 == 0:
            print('On row ',i, 'dir', u,v)
        for j in width_range:
            data_costs =data_costs_cache.get_data_costs(i,j)
            directional_regularizers = regularizer_cache.get_regularization_penalty((i-u,j-v),(u,v))
            if directional_regularizers is not None:
                directional_penalties = np.add(data_costs,directional_regularizers)
            else:
                directional_penalties = data_costs

            #Update final costs
            directional_penalty[i,j,:,:] = directional_penalties

            #Compute regularization penalties 

            new_directional_regularizers = regularizer.compute_regularizers(directional_penalties)
            
            regularizer_cache.add_regularization_penalty((i,j),(u,v),new_directional_regularizers)
        
        regularizer_cache.clear_keys_if_last_use((i,j),[(u,v)])
    return directional_penalty
