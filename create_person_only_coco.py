import fiftyone as fo
import fiftyone.zoo as foz
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--export-dir", type=str, required=True)
    args = parser.parse_args()
    # max_samplesを指定しないとなぜかperson以外のデータも読み込んでしまうので
    # 適当に十分大きい数字を指定している
    train_dataset = foz.load_zoo_dataset("coco-2017", split="train",
            label_types=["detections", "segmentations"] ,classes="person",
            only_matching=True, max_samples=250000)
    valid_dataset = foz.load_zoo_dataset("coco-2017", split="validation",
            label_types=["detections", "segmentations"] ,classes="person",
            only_matching=True, max_samples=5000)

    train_dataset.export(
        export_dir=os.path.join(args.export_dir, "train2017_person_only"),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="segmentations",
    )

    valid_dataset.export(
        export_dir=os.path.join(args.export_dir, "val2017_person_only"),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="segmentations",
    )

if __name__ == "__main__":
    main()
