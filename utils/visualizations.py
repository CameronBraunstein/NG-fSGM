from itypes import Dataset
import numpy as np
import torchvision.transforms as transforms


def view_torch_tensor(img):
    sample = transforms.ToPILImage()(img)
    sample.show()

def visualize_flow(args,dataloader,flow_prediction):
    imfile0,imfile1, gt_flow_file = dataloader.get_paths()
    output_name = '{}_{}'.format(args.dataset,args.scene)
    iviz_file = 'computed_flow_scenes/{}.json'.format(output_name)
    ds = Dataset(file=iviz_file, auto_write=True)
    with ds.viz.new_row() as row:
        row.add_cell("image", var="im0")
        row.add_cell("image", var="im1")
        row.add_cell("flow",  var="predicted_flow")
        row.add_cell("flow", var= "gt_flow")
    with ds.seq.group(output_name, label=args.scene) as group:
        with group.item("item_{}".format(args.scene), label="Item {}".format(args.scene)) as item:
            data_0,data_1 = dataloader.get_frames()
            item["im0"].set_data(np.asarray(data_0), dims="hwc") 
            item["im1"].set_data(np.asarray(data_1), dims="hwc")
            item["predicted_flow"].set_data(flow_prediction, dims="hwc")
            item["gt_flow"].set_ref(gt_flow_file, rel_to="cwd")
    return iviz_file


if __name__ == '__main__':
    scene_name = 'RubberWhale'
    iviz_file = 'computed_flow_scenes/gt_{}.json'.format(scene_name)
    ds = Dataset(file=iviz_file, auto_write=True)
    with ds.viz.new_row() as row:
        row.add_cell("image", var="im0")
        row.add_cell("image", var="im1")
        row.add_cell("flow",  var="Flow")
    with ds.seq.group('Ground truth', label=scene_name) as group:
        with group.item("item_{}".format(scene_name), label="Item {}".format(scene_name)) as item:
            #item["im0"].set_ref(imfile0, rel_to="cwd") 
            #item["im1"].set_ref(imfile1, rel_to="cwd")
            item["im0"].set_ref('Middlebury/frames/{}/frame10.png'.format(scene_name), rel_to="cwd") 
            item["im1"].set_ref('Middlebury/frames/{}/frame11.png'.format(scene_name), rel_to="cwd")
            #item["Flow"].set_ref('Middlebury/gt-flow/{}/flow10.flo'.format(scene_name), rel_to="cwd")
            item["Flow"].set_ref('RubberWhale.flo', rel_to="cwd")
    print('Run:')
    print('iviz',iviz_file)