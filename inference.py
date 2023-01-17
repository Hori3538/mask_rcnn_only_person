# from utils.engine import  evaluate
from engine import evaluate
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
    parser.add_argument("--val-images-root-path", type=str, default="~/person_only_coco/val2017_person_only/data")
    # parser.add_argument("--val_images-root-path", type=str, default="~/person_only_coco/train2017_person_only/data")
    parser.add_argument("--val-json-annotation-path", type=str, default="/home/amsl/person_only_coco/val2017_person_only/labels.json")
    # parser.add_argument("--val-json-annotation-path", type=str, default="/home/amsl/person_only_coco/train2017_person_only/labels.json")
    parser.add_argument("--label-file-path", type=str, default="./object_detection_classes_coco.txt")
    parser.add_argument("--colors-file-path", type=str, default="./colors.txt")
    parser.add_argument("--is-custom", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--weight-path", type=str, default="./model.pth")
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

    dataset_val = CustomDataset(root=args.val_images_root_path, annFile=args.val_json_annotation_path, transforms=CustomDataset.get_transform(False), is_custom=args.is_custom)
    
    indices_val = []
    for i in range(len(dataset_val)):
        if i != 2465:
            indices_val.append(i)
    dataset_val = torch.utils.data.Subset(dataset_val, indices_val)

    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
        )

    num_classes = args.num_classes
    # model = models.detection.maskrcnn_resnet50_fpn_v2(num_classes=2)
    # model = models.detection.maskrcnn_resnet50_fpn(num_classes=2)
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=torch.device(device)))


    evaluate(model, data_loader_val, device=device)
    for image, target in data_loader_val:
        # print(f"image tensor shape: {image.shape}")

        image = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_original = image.copy()

        object_num = target[0]['labels'].nelement()
        boxes = target[0]['boxes']
        labels = target[0]['labels']
        masks = target[0]['masks']
        for i in range(object_num):
            rect: List[int] = [int(data) for data in boxes[i].tolist()]
            id: int = labels[i]
            mask = masks[i].numpy() * 255
            drawer.draw_bbox(image, image_original, mask, rect, id)

        # cv2.imshow("image", image)
        # key = cv2.waitKey(0)
        # if key == ord("q") or key == ord("c"):
        #     break
        # cv2.destroyAllWindows()

    # data_loader_test = torch.utils.data.DataLoader(
                    # dataset_val, batch_size=4, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    # model_path = 'model.pth'
    # torch.save(model.state_dict(), model_path)

    # for image, target in data_loader_test:
    #
    #     image = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     image_original = image.copy()
    #
    #     object_num = target[0]['labels'].nelement()
    #     boxes = target[0]['boxes']
    #     labels = target[0]['labels']
    #     masks = target[0]['masks']
    #     for i in range(object_num):
    #         rect: List[int] = [int(data) for data in boxes[i].tolist()]
    #         id: int = labels[i]
    #         mask = masks[i].numpy() * 255
    #         drawer.draw_bbox(image, image_original, mask, rect, id)
    #
    #     cv2.imshow("image", image)
    #     key = cv2.waitKey(0)
    #     if key == ord("q") or key == ord("c"):
    #         break
    #     cv2.destroyAllWindows()
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
