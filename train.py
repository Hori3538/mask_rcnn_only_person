from torch.optim import lr_scheduler
# from utils.engine import train_one_epoch, evaluate
from engine import train_one_epoch, evaluate
# import utils.utils
import transforms as T
import utils
import torchvision.models as models
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import cv2
import numpy as np
from typing import List
from dataset import CustomDataset, Drawer 

# import sys
# sys.path.append("./utils")
# print(sys.path)
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_images-root-path", type=str, default="~/person_only_coco/train2017_person_only/data")
    # parser.add_argument("--train_images-root-path", type=str, default="~/coco_data/train2017")
    parser.add_argument("--train_json-annotation-path", type=str, default="/home/amsl/person_only_coco/train2017_person_only/labels.json")
    # parser.add_argument("--train_json-annotation-path", type=str, default="/home/amsl/coco_data/annotations/instances_train2017.json")
    parser.add_argument("--val_images-root-path", type=str, default="~/person_only_coco/val2017_person_only/data")
    # parser.add_argument("--val_images-root-path", type=str, default="~/coco_data/val2017")
    parser.add_argument("--val_json-annotation-path", type=str, default="/home/amsl/person_only_coco/val2017_person_only/labels.json")
    # parser.add_argument("--val_json-annotation-path", type=str, default="/home/amsl/coco_data/annotations/instances_val2017.json")
    parser.add_argument("--label-file-path", type=str, default="./object_detection_classes_coco.txt")
    parser.add_argument("--colors-file-path", type=str, default="./colors.txt")
    parser.add_argument("--is-custom", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()

    # fiftyoneをインストールしたあたりから動かなくなった
    # class_names = np.loadtxt(args.label_file_path, dtype='str', delimiter='\n')
    class_names = []
    with open(args.label_file_path) as f:
        while True:
            label = f.readline().rstrip()
            if label:
                class_names.append(label)
            else:
                break
    colors = np.loadtxt(args.colors_file_path, dtype='int', delimiter=' ')

    drawer = Drawer(class_names, colors)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train = CustomDataset(root=args.train_images_root_path, annFile=args.train_json_annotation_path, transforms=CustomDataset.get_transform(True), is_custom=args.is_custom)
    dataset_val = CustomDataset(root=args.val_images_root_path, annFile=args.val_json_annotation_path, transforms=CustomDataset.get_transform(False), is_custom=args.is_custom)
    
    indices_train = torch.randperm(len(dataset_train)).tolist()
    # indices_val = torch.randperm(len(dataset_val)).tolist()
    # indices_val = [n for n in range(2465, len(dataset_val))]
    indices_val = []
    # for i in range(len(dataset_val)):
    #     if i != 2465:
    #         indices_val.append(i)
    # dataset_train = torch.utils.data.Subset(dataset_train, indices_train[:1000])
    dataset_val = torch.utils.data.Subset(dataset_val, indices_val)

    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=4, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
        )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=8, shuffle=False, num_workers=0, collate_fn=utils.collate_fn
        )

    num_classes = args.num_classes
    # model = models.detection.maskrcnn_resnet50_fpn_v2(num_classes=2)
    # model = models.detection.maskrcnn_resnet50_fpn(num_classes=2)
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.01)

    num_epochs = 5

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    data_loader_test = torch.utils.data.DataLoader(
                    dataset_val, batch_size=4, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    # model.eval() 
    #
    #     # Let's create a dummy input tensor  
    # dummy_input = torch.randn(1, 3, 244, 244)  
    #
    # # Export the model   
    # torch.onnx.export(model,         # model being run 
    #      dummy_input,       # model input (or a tuple for multiple inputs) 
    #      "ImageClassifier.onnx",       # where to save the model  
    #      export_params=True,  # store the trained parameter weights inside the model file 
    #      opset_version=10,    # the ONNX version to export the model to 
    #      do_constant_folding=True,  # whether to execute constant folding for optimization 
    #      input_names = ['modelInput'],   # the model's input names 
    #      output_names = ['modelOutput'], # the model's output names 
    #      dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
    #                             'modelOutput' : {0 : 'batch_size'}}) 
    # print(" ") 
    # print('Model has been converted to ONNX') 

if __name__ == "__main__":
    main()
