
class fSGM:
    def __init__(self,args):
        self.census_window = args.census_window
        self.path_directions = args.path_directions
        self.P1 = args.P1
        self.P2 = args.P2

    def compute_flow(self,frame_0,frame_1):
        #  'It first computes pixel-wise matching costs of corresponding pixels in two 
        #  frames for all disparities in the search space. '
        self.matching_costs = self.compute_matching_costs(frame_0,frame_1)

        return None

    def compute_matching_costs(self,frame_0,frame_1):
        print('HELLO')
        return None
