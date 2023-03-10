import argparse
from dataloader import Dataloader
from NG_fSGM import NG_fSGM
from fSGM import fSGM


parser = argparse.ArgumentParser(description='Neighbor Guided flow-Semi Global Matching.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset',choices=['Middlebury'],
                    help='evaluation data set')
parser.add_argument('--scene',choices=['Beanbags','Dimetrodon'],
                    help='evaluation data set scene')

parser.add_argument('--algorithm',choices=['NG-fSGM','fSGM'], default='NG-fSGM',
                    help='Choice of optical flow algorithm')

#See IV. Results, paragraph 2 for defaults which give optimal performance
parser.add_argument('--census_window',default=11, type=int,
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


def main():
    args = parser.parse_args()
    dataloader = Dataloader(args)
    frame_0,frame_1 =dataloader.getframes()
    if (args.algorithm == 'NG-fSGM'):
        predictor = NG_fSGM(args)
    elif (args.algorithm == 'fSGM'):
        predictor = fSGM(args)
    
    flow_prediction = predictor.compute_flow(frame_0,frame_1)
    

if __name__ == '__main__':
    main()