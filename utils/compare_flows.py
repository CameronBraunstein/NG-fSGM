import argparse
from file_io import get_as_numpy_tensor
from itypes import Dataset
import numpy as np


def calculate_diffs(gt_flow,predicted_flow):
    flow_diff = gt_flow-predicted_flow
    height,width,_ = gt_flow.shape
    max_u = 0
    max_v = 0
    coded_flow_diff = np.zeros((height,width,3))
    for i in range(height):
        for j in range(width):
            if abs(gt_flow[i,j,0]) > height or abs(gt_flow[i,j,1])>width:
                coded_flow_diff[i,j,:] = [0,255,0]
                continue

            if abs(gt_flow[i,j,0]) > abs(max_v):
                max_v = gt_flow[i,j,0]
            if abs(gt_flow[i,j,1]) > abs(max_u):
                max_u = gt_flow[i,j,1]

            if abs(flow_diff[i,j,0]) > 1.5 or abs(flow_diff[i,j,1]) > 1.5:
                coded_flow_diff[i,j,:] = [255,0,255]
            elif abs(flow_diff[i,j,0]) > 0.5 or abs(flow_diff[i,j,1]) > 0.5:
                coded_flow_diff[i,j,:] = [255,0,0]
    print('max u', max_u, 'max_v', max_v)
    return flow_diff, coded_flow_diff

def main(args):
    gt_flow = get_as_numpy_tensor(args.ground_truth)
    output_name = 'RubberWhale'
    iviz_file = 'flow_comparisons/{}.json'.format(output_name)
    ds = Dataset(file=iviz_file, auto_write=True)
    with ds.viz.new_row() as row:
        row.add_cell("flow",  var= "gt_flow")
        row.add_cell("flow",  var="predicted_flow")
        row.add_cell("flow",  var="flow_diff")
        row.add_cell("image", var="coded_flow_diff")
    with ds.seq.group(output_name, label=output_name) as group:
        for file in args.flow_files:
            predicted_flow = get_as_numpy_tensor(file)
            flow_diff, coded_flow_diff = calculate_diffs(gt_flow,predicted_flow)
            with group.item("item_{}".format(file), label="Item {}".format(file)) as item:
                item["gt_flow"].set_data(gt_flow, dims="hwc")
                item["predicted_flow"].set_data(predicted_flow, dims="hwc")
                item["flow_diff"].set_data(flow_diff, dims="hwc")
                item["coded_flow_diff"].set_data(coded_flow_diff, dims="hwc")

    print('iviz', iviz_file)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow Comparison Utility.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ground_truth',
                    help='path from project directory to ground truth flow')
    parser.add_argument('--flow_files', nargs='+',
                        help='generated flow file(s) to compare against the ground truth')
    args = parser.parse_args()
    main(args)