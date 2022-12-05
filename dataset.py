from torchvision.datasets import CocoDetection
import torchvision
import torch
# import utils.transforms as T
import transforms as T
# import utils.utils
import utils

class CustomDataset(CocoDetection):
    def __getitem__(self, index):
        #継承した親メソッドから必要な変数を受け取る
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        for i, t in enumerate(target):
            if not t['segmentation']:
                del target[i]

        #target(アノテーションデータ)をデータセットとして読み込めるように変換する      
        boxes = [x['bbox'] for x in target]
        labels = [x['category_id'] for x in target]
        image_id = idx
        area = [box[2] * box[3] for box in boxes]
        iscrowd = [x['iscrowd'] for x in target]

        masks = [self.coco.annToMask(x) for x in target]

        targets = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets["boxes"] = torchvision.ops.box_convert(boxes,'xywh','xyxy') if boxes.nelement()!=0 else []
        targets["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        masks = np.array(masks)
        targets["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        targets["image_id"] = torch.tensor([image_id])
        targets["area"] = torch.tensor(area)
        targets["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # PIL -> tensor
        # image, targets = T.PILToTensor()(image, targets)

        #transformsを受け取っていれば内容に従って変換する
        if self.transforms is not None:
        # if self.transforms is not None and len(labels) != 0:
            image, targets = self.transforms(image, targets)

        return image, targets

    @classmethod
    def get_transform(cls, train: bool) -> T.Compose:
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

from torch.utils.data import DataLoader
from argparse import ArgumentParser
import cv2
import numpy as np
from typing import List


def test() -> None:
    parser = ArgumentParser()
    # parser.add_argument("--images-root-path", type=str, default="~/person_only_coco/val2017_person_only/data")
    parser.add_argument("--images-root-path", type=str, default="~/person_only_coco/train2017_person_only/data")
    # parser.add_argument("--json-annotation-path", type=str, default="/home/amsl/person_only_coco/val2017_person_only/labels.json")
    parser.add_argument("--json-annotation-path", type=str, default="/home/amsl/person_only_coco/train2017_person_only/labels.json")
    parser.add_argument("--label-file-path", type=str, default="./object_detection_classes_coco.txt")
    parser.add_argument("--colors-file-path", type=str, default="./colors.txt")
    parser.add_argument("--is-custom", type=bool, default=True)
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

    drawer = Drawer(class_names, colors, args.is_custom)

    dataset = CustomDataset(root=args.images_root_path, annFile=args.json_annotation_path,
            transforms=CustomDataset.get_transform(True))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=utils.collate_fn)

    print(f"datal len: {len(dataset)}")
    for image, target in dataloader:

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

        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"):
            break
        cv2.destroyAllWindows()

class Drawer():
    def __init__(self, class_names, colors, is_custom) -> None:
        self._class_names = class_names
        self._colors = colors
        self._is_custom = is_custom

    def draw_bbox(self, image: np.ndarray, image_original: np.ndarray, mask, rect: List[int], id: int) -> None:
        cv2.rectangle(image, rect[:2], rect[2:], (255, 255, 255), 2)
        class_name = self._class_names[id] if self._is_custom else self._class_names[id-1]
        cv2.putText(image, class_name, rect[:2],
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        color = self._colors[id %  len(self._colors)]
        colored_roi = (0.3 * color + 0.7 * image_original).astype(np.uint8)
        image[:] = np.where(mask[:, :, np.newaxis] == 0, image, colored_roi)

if __name__ == "__main__":
    test()
