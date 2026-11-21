import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CelebAMTLDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        tasks=None,
        image_size=224,
        train=True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.tasks = list(tasks)

        partition_file = os.path.join(data_dir, "Eval", "list_eval_partition.txt")
        attr_file = os.path.join(data_dir, "Anno", "list_attr_celeba.txt")
        self.img_dir = os.path.join(data_dir, "Img", "img_align_celeba")

        split_to_id = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        split_id = split_to_id[split]

        with open(partition_file, "r") as f:
            partitions = [line.strip().split() for line in f if line.strip()]

        partition_map = {
            name: int(part)
            for name, part in partitions
        }

        with open(attr_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        attr_names = lines[1].split()
        task_to_idx = {name: i for i, name in enumerate(attr_names)}

        missing = [t for t in self.tasks if t not in task_to_idx]
        if missing:
            raise ValueError(f"Unknown CelebA attributes: {missing}")

        self.img_names = []
        self.labels = []

        for line in lines[2:]:
            parts = line.split()
            img_name = parts[0]

            if partition_map[img_name] != split_id:
                continue

            raw_attrs = parts[1:]
            label_dict = {
                task: 1.0 if raw_attrs[task_to_idx[task]] == "1" else 0.0
                for task in self.tasks
            }

            self.img_names.append(img_name)
            self.labels.append(label_dict)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        labels = {
            task: torch.tensor(self.labels[index][task], dtype=torch.float32)
            for task in self.tasks
        }

        return img, labels

    def __len__(self):
        return len(self.img_names)