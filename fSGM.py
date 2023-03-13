from utils import census_transform, hamming_distance, DataCostCalculator, DirectionalRegularizerCache, calculate_Potts_directional_regularizers, get_offset_from_window_size
import torch

#Based on description in Hierarchical Scan-Line Dynamic Programming for 
# Optical Flow Using Semi-Global Matching, Simon Hermann & Reinhard Klette

class fSGM:
    def __init__(self,args):
        self.displacement_window = args.displacement_window
        self.path_directions = args.path_directions
        self.P1 = args.P1
        self.P2 = args.P2
        self.penalty_model = args.penalty_model

    def compute_flow(self,frame_0,frame_1):
        height,width = frame_0.size()[1:3]        

        data_cost_calc = DataCostCalculator(frame_0,frame_1,self.displacement_window)


        self.total_flow_costs = torch.zeros(height,width,self.displacement_window,self.displacement_window)
        #Forward pass
        self.forward_or_backward_pass(height,width,data_cost_calc,is_forward=True)
        #Backward pass
        self.forward_or_backward_pass(height,width,data_cost_calc,is_forward=False)

        self.flow = torch.zeros(height,width,2)
        self.get_flows_from_total_costs(height,width)
        print(self.flow)



        #print(self.total_flow_costs)

        # for i in range(height):
        #     for j in range(width):
        #         base_census = census_0[i,j]
        #         displacement_costs = torch.zeros(self.census_window,self.census_window)
        #         for k in range(self.census_window):
        #             for l in range(self.census_window):
        #                 displacement_costs[k,l] = hamming_distance(base_census,census_1[i+k,j+l])
        #         #print(displacement_costs)
        #         for (u,v) in [(1,0),(1,1),(0,1),(1,1)]:
        #             directional_regularizers = regularizer_cache.get_regularization_penalty((i-u,j-v),(u,v))
        #             if directional_regularizers is not None:
        #                 directional_penalties = torch.add(displacement_costs,directional_regularizers)
        #             else:
        #                 directional_penalties = displacement_costs

        #             self.total_flow_costs[i,j,:,:] = torch.add(self.total_flow_costs[i,j,:,:],directional_penalties)

        #             #Compute regularization penalties
        #             if self.penalty_model == 'Potts':
        #                 new_directional_regularizers = calculate_Potts_directional_regularizers(directional_penalties,self.P1,self.P2)
        #             else:
        #                 raise Exception("Only supporting Potts model for now")
                    
        #             regularizer_cache.add_regularization_penalty((i,j),(u,v),new_directional_regularizers)
                
        #         regularizer_cache.clear_keys_if_last_use((i,j))



        return self.flow
    
    def get_flows_from_total_costs(self,height,width):
        window_offset = get_offset_from_window_size(self.displacement_window)
        for i in range(height):
            for j in range(width):
                argmin = torch.argmin(self.total_flow_costs[i,j,:,:])
                self.flow[i,j,0] = (argmin // self.displacement_window) -window_offset
                self.flow[i,j,1] = (argmin % self.displacement_window) -window_offset

    
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
            for j in width_range:
                data_costs =data_cost_calc.get_data_costs(i,j)

                for (u,v) in direction_vectors:
                    directional_regularizers = regularizer_cache.get_regularization_penalty((i-u,j-v),(u,v))
                    if directional_regularizers is not None:
                        directional_penalties = torch.add(data_costs,directional_regularizers)
                    else:
                        directional_penalties = data_costs

                    #Update final costs
                    self.total_flow_costs[i,j,:,:] = torch.add(self.total_flow_costs[i,j,:,:],directional_penalties)

                    #Compute regularization penalties 
                    if self.penalty_model == 'Potts':
                        new_directional_regularizers = calculate_Potts_directional_regularizers(directional_penalties,self.P1,self.P2)
                    else:
                        raise Exception("Only supporting Potts model for now")
                    
                    regularizer_cache.add_regularization_penalty((i,j),(u,v),new_directional_regularizers)
                
                regularizer_cache.clear_keys_if_last_use((i,j),direction_vectors)
        return
