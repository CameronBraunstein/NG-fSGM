

import numpy as np


class RegularizationModel:
    def __init__(self,args):
        self.P1 = args.P1
        self.P2 = args.P2
        self.penalty_model = args.penalty_model
    def compute_regularizers(self,directional_penalties):
        if self.penalty_model == 'Potts':
            return calculate_Potts_directional_regularizers(directional_penalties,self.P1,self.P2)
        else:
            raise Exception("Only supporting Potts model for now")


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