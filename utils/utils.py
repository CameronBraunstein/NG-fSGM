#TODO: Move all logic into appropriately named files
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
    
def get_offset_from_window_size(window_size):
    return window_size //2
