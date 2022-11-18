from torchvision.datasets import CocoDetection
import torchvision
import torch
import utils.transforms as T
import utils.utils

class CustomDataset(CocoDetection):
    def __getitem__(self, index):
        #継承した親メソッドから必要な変数を受け取る
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        # print(f"target: {target}")

        #target(アノテーションデータ)をデータセットとして読み込めるように変換する      
        boxes = [x['bbox'] for x in target]
        # print(f"boxes: {boxes}\n")
        labels = [x['category_id'] for x in target]
        # print(f"labels: {labels}")
        image_id = idx
        area = [box[2] * box[3] for box in boxes]
        iscrowd = [x['iscrowd'] for x in target]
        masks = [self.coco.annToMask(x) for x in target]

        targets = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # print(f"boxes tensor: {boxes}")
        # print(f"boxes.nelement: {boxes.nelement()}")
        targets["boxes"] = torchvision.ops.box_convert(boxes,'xywh','xyxy') if boxes.nelement()!=0 else []
        # print(f"converted boxes: {targets['boxes']}")
        targets["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        targets["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        targets["image_id"] = torch.tensor([image_id])
        targets["area"] = torch.tensor(area)
        targets["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        #transformsを受け取っていれば内容に従って変換する
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets

def get_transform(train: bool) -> T.Compose:
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def test() -> None:
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--images-root-path", type=str)
    parser.add_argument("--json-annotation-path", type=str)
    args = parser.parse_args()

    dataset = CustomDataset(root=args.images_root_path, annFile=args.json_annotation_path,
            transforms=get_transform(train=True))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=utils.utils.collate_fn)
    dataset[0]

    # images, targets = next(iter(dataloader))
    for image, target in dataloader:
        print(f"shape of images: {image[0].shape}")
        # print(f"type of images: {type(image)}")
        print(image)
        print(f"shape of targets: {target[0]}\n")
        # print(f"type of targets: {type(target)}\n")
        print(target)
        pass

if __name__ == "__main__":
    test()
