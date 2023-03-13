import argparse
from dataloader import Dataloader
from NG_fSGM import NG_fSGM
from fSGM import fSGM
from utils.visualizations import visualize_flow
from utils.file_io import save_cost_tensor,save_flow_file
import timeit


parser = argparse.ArgumentParser(description='Neighbor Guided flow-Semi Global Matching.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset',choices=['Middlebury'],
                    help='evaluation data set')
parser.add_argument('--scene',choices=['Beanbags','Dimetrodon','DogDance','Grove2','RubberWhale'],
                    help='evaluation data set scene')

parser.add_argument('--algorithm',choices=['NG-fSGM','fSGM'], default='NG-fSGM',
                    help='Choice of optical flow algorithm')

parser.add_argument('--penalty_model',choices=['Potts'], default='Potts',
                    help='Choice of penalty function')

#See IV. Results, paragraph 2 for defaults which give optimal performance
parser.add_argument('--displacement_window',default=11, type=int,
                    help='dimension of the window with in which we search for optical flow')

parser.add_argument('--path_directions',default=8, type=int,
                    help='number of local directions to search through when aggregating flow costs')

parser.add_argument('--P1',default=20, type=int,
                    help='regularization penalty for small changes in flow field (P1 <= P2)')

parser.add_argument('--P2',default=60, type=int,
                    help='regularization penalty for large changes in flow field (P1 <= P2)')

parser.add_argument('--N',default=3, type=int,
                    help='Number of best flow vectors of neighbor which are considered (plus the surrounding neighborhood)')

parser.add_argument('--M',default=3, type=int,
                    help='Number of random flow vectors which are considered')

parser.add_argument("--visualize_flow", action='store_true',
                    help="Add to make a visualization using itypes")

parser.add_argument("--save_cost_tensor", action='store_true',
                    help="Store the total cost tensor for further processes. The standard user will not need to store this intermediate step")

parser.add_argument("--save_flow_file", action='store_true',
                    help="Save the flow file in .flo format")
  


def main():
    args = parser.parse_args()
    dataloader = Dataloader(args)
    frame_0,frame_1 =dataloader.get_frames()
    if (args.algorithm == 'NG-fSGM'):
        predictor = NG_fSGM(args)
    elif (args.algorithm == 'fSGM'):
        predictor = fSGM(args)
    
    start_time = timeit.default_timer()
    flow_prediction = predictor.compute_flow(frame_0,frame_1)
    print("Completed flow prediction in:", timeit.default_timer() - start_time)
    if args.visualize_flow:
        iviz_file = visualize_flow(args,dataloader,flow_prediction)
        print()
        print('To view your output, run:')
        print('iviz',iviz_file)
        print()
    if args.save_cost_tensor:
        path_name = save_cost_tensor(args,predictor)
        print()
        print('Saved cost tensor at ', path_name)
        print()
    if args.save_flow_file:
        path_name = save_flow_file(args,flow_prediction)
        print()
        print('Saved calculated flow at ', path_name)
        print()


    

if __name__ == '__main__':
    main()