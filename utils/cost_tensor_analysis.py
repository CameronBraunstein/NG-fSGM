import argparse
from file_io import get_as_numpy_tensor
import numpy as np
from utils import get_offset_from_window_size
import matplotlib.pyplot as plt

def show_histogram(close_to_gt_scores):
    fig = plt.figure()
    plt.bar(range(len(close_to_gt_scores),0,-1),np.flip(close_to_gt_scores), color ='maroon',)
    plt.xlabel("Candidate Ranking")
    plt.ylabel("Frequency")
    plt.title("Frequency of Candidate Displacement within a pixel of being correct")
    plt.show()

def show_drift(differences):
    drifts = [np.average(i) for i in differences]
    fig = plt.figure()
    plt.bar(range(len(drifts),0,-1),np.flip(drifts), color ='maroon',)
    plt.xlabel("Candidate Ranking")
    plt.ylabel("Average Drift of next lower candidate")
    plt.title("Average drift between candidates")
    plt.show()


def main(args):
    gt_flow = get_as_numpy_tensor(args.ground_truth)
    cost_tensor = get_as_numpy_tensor(args.cost_tensor)
    height,width,displacement_window = cost_tensor.shape[0:3]
    window_offset = get_offset_from_window_size(displacement_window)
    close_to_gt_scores = np.zeros(args.candidates)
    differences =[[] for i in range(args.candidates -1 )]
    print(differences)
    occlusion_count = 0
    for i in range(height):
        for j in range(width):
            if (abs(gt_flow[i,j,0]) > width) or (abs(gt_flow[i,j,1]) > height): #occlusion
                occlusion_count = occlusion_count +1
                continue 
            indices = np.argsort(cost_tensor[i,j,:,:].flatten())[:args.candidates]
            for index,ranking in zip(indices,range(len(indices))):
                flow_u = (index % displacement_window)-window_offset
                flow_v = (index // displacement_window)-window_offset
                if (abs(gt_flow[i,j,0]-flow_u) <= 0.5) and (abs(gt_flow[i,j,1]-flow_v) <= 0.5):
                    close_to_gt_scores[ranking] = close_to_gt_scores[ranking]+1
                    break
                if ranking > 0:
                    differences[ranking-1].append((flow_u-old_flow_u)**2 + (flow_v-old_flow_v)**2)
                old_flow_u = flow_u
                old_flow_v = flow_v
    print(close_to_gt_scores, height*width-sum(close_to_gt_scores)-occlusion_count)
    show_histogram(close_to_gt_scores)
    show_drift(differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cost Tensor Analysis Utility.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ground_truth',
                    help='path from project directory to ground truth flow')
    parser.add_argument('--cost_tensor',
                        help='path to saved cost tensor as .npy file')
    parser.add_argument('--candidates',type=int,
                        help='number of top candidates to examine for metrics')
    args = parser.parse_args()
    main(args)