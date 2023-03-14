import os.path
from PIL import Image
import numpy as np

def save_flow_file(args,flow_prediction):
    u = flow_prediction[:,:,0]
    v = flow_prediction[:,:,1]
    height, width = u.shape

    file_name = '{}.flo'.format(args.scene)
    f = open(file_name, 'wb')
    # write the header
    np.array([202021.25]).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * 2))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
    return file_name

def load_flow_file(file_name):
    with open(file_name,'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise Exception('Magic number incorrect. Invalid .flo file')
        else:
            width = np.fromfile(f, np.int32, count=1)
            height = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(width)*int(height))
            data_resize = np.resize(data, (int(height), int(width), 2))
            return data_resize


def save_cost_tensor(args,predictor):
    path_name = '{}_cost_tensor.npy'.format(args.scene)
    cost_tensor = getattr(predictor,'total_flow_costs')
    with open(path_name, 'wb') as f:
        np.save(f, cost_tensor)
    return path_name

def load_cost_tensor(path_name):
    with open(path_name, 'rb') as f:
        cost_tensor = np.load(f)
        return cost_tensor


# def get_torch_tensor(path_name):
#     _ , extension =  os.path.splitext(path_name)
#     if extension == '.png':
#         img = Image.open(path_name)
#         convert_tensor = transforms.ToTensor()
#         return convert_tensor(img)
    
def get_as_numpy_tensor(path_name):
    _ , extension =  os.path.splitext(path_name)
    if extension == '.png':
        im_frame = Image.open(path_name)
        np_array = np.array(im_frame)
        print(np_array.shape)
        return np_array

def get_as_PIL_image(path_name):
    _ , extension =  os.path.splitext(path_name)
    if extension == '.png':
        im_frame = Image.open(path_name)
        return im_frame