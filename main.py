import argparse
from dataloader import Dataloader


parser = argparse.ArgumentParser(description='Neighbor Guided flow-Semi Global Matching.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset',choices=['Middlebury'],
                    help='evaluation data set')
parser.add_argument('--scene',choices=['Beanbags','Dimetrodon'],
                    help='evaluation data set scene')


def main():
    args = parser.parse_args()
    dataloader = Dataloader(args)
    frame_0,frame_1 =dataloader.getframes()
    print(frame_0.size())
    

if __name__ == '__main__':
    main()