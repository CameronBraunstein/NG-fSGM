import argparse
from file_io import get_as_numpy_tensor



def main(args):
    gt_flow = get_as_numpy_tensor(args.ground_truth)
    cost_tensor = get_as_numpy_tensor(args.cost_tensor)
    print(cost_tensor.shape)
    #TO DO: Analysis- how close are the top n candidates?
    #How often is the pixel wise closest candidate in the correct set of candidates?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cost Tensor Analysis Utility.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ground_truth',
                    help='path from project directory to ground truth flow')
    parser.add_argument('--cost_tensor',
                        help='path to saved cost tensor as .npy file')
    args = parser.parse_args()
    main(args)