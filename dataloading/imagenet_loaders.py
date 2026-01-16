import argparse
import os
import io
import time
import pickle

import h5py
import pyarrow as pa

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

from PIL import Image

import json

class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):

        file_name = "imagenet_train.pkl" if split == "train" else "imagenet_val.pkl"
        self.data_root = root
        
        with open(os.path.join(self.data_root, file_name), "rb") as f:
            train_data = pickle.load(f)

        self.samples = list(train_data.keys())
        self.targets = list(train_data.values())
        self.transform = transform

    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.data_root, self.samples[idx])).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]    

class ImageNetH5(Dataset):
    def __init__(self, train_data_path, split, transform=None):
        self.train_data_path = train_data_path
        self.split = split
        self.h5file = None
        self.imgs = None
        self.targets = None
        with h5py.File(train_data_path, 'r') as h5file:
            self._len = len(h5file[split]["images"])
        self.transform = transform

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        if self.h5file is None:
            self.h5file = h5py.File(self.train_data_path, 'r')[self.split]
            self.imgs = self.h5file["images"]
            self.targets = self.h5file["targets"]
        img_string = self.imgs[idx]
        target = self.targets[idx]

        with io.BytesIO(img_string) as byte_stream:
            with Image.open(byte_stream) as img:
                img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


class ImageNetArrow(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform

        self.arrowfile = None
        self.reader = None
        with pa.OSFile(self.data_root, 'rb') as f:
            with pa.ipc.open_file(f) as reader:
                self._len = reader.num_record_batches

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.arrowfile is None:
            self.arrowfile = pa.OSFile(self.data_root, 'rb')
            self.reader = pa.ipc.open_file(self.arrowfile)

        row = self.reader.get_batch(idx)
        img_string = row['image_data'][0].as_py()
        target = row['label'][0].as_py()

        with io.BytesIO(img_string) as byte_stream:
            with Image.open(byte_stream) as img:
                img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


def main(args):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])

    if args.dset_type == 'fs':
        image_datasets = ImageNet(args.data_root, "train", transform)
    elif args.dset_type == 'h5':
        image_datasets = ImageNetH5(args.data_root, "train", transform)
    elif args.dset_type == 'arrow':
        image_datasets = ImageNetArrow(args.data_root, transform)
    else:
        assert False

    sampler = DistributedSampler(
        image_datasets,
        num_replicas=int(os.getenv('SLURM_NTASKS')),
        rank=int(os.getenv('SLURM_PROCID')),
        shuffle=args.shuffle,
    )

    dataloader = DataLoader(
        image_datasets,
        batch_size=128,
        num_workers=int(os.getenv('SRUN_CPUS_PER_TASK')),
        pin_memory=True,
        sampler=sampler,
    )

    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()

    td = (end_time - start_time)
    print(time.strftime("%H:%M:%S", time.gmtime(td)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-d', required=True)
    parser.add_argument('--dset_type', choices=['fs', 'h5', 'arrow'])
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    main(args)
