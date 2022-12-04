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

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_images-root-path", type=str, default="~/person_only_coco/train2017_person_only/data")
    parser.add_argument("--train_json-annotation-path", type=str, default="/home/amsl/person_only_coco/train2017_person_only/labels.json")
    parser.add_argument("--val_images-root-path", type=str, default="~/person_only_coco/val2017_person_only/data")
    parser.add_argument("--val_json-annotation-path", type=str, default="/home/amsl/person_only_coco/val2017_person_only/labels.json")
    parser.add_argument("--label-file-path", type=str, default="./object_detection_classes_coco.txt")
    parser.add_argument("--colors-file-path", type=str, default="./colors.txt")
    parser.add_argument("--is-custom", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # transform = T.RandomHorizontalFlip(0.5)
    dataset_train = CustomDataset(root=args.train_images_root_path, annFile=args.train_json_annotation_path, transforms=CustomDataset.get_transform(True))
    dataset_val = CustomDataset(root=args.val_images_root_path, annFile=args.val_json_annotation_path, transforms=CustomDataset.get_transform(False))
    
    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
        )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
        )

    model = models.detection.maskrcnn_resnet50_fpn_v2(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.01)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

if __name__ == "__main__":
    main()
