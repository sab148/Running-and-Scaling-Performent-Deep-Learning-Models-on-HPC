import argparse
import json
import os
from tqdm import tqdm
import pyarrow as pa
import h5py
import numpy as np
import pickle

def save_files(args):

    splits = ["train", "val"]

    with open(os.path.join(args.data_root, "imagenet_train.pkl"), "rb") as f:
        train_data = pickle.load(f)

    train_samples = list(train_data.keys())
    train_targets = list(train_data.values())

    with open(os.path.join(args.data_root, "imagenet_val.pkl"), "rb") as f:
        val_data = pickle.load(f)

    val_samples = list(val_data.keys())
    val_targets = list(val_data.values())

    print(f"Train samples: {len(train_samples)}")
    print(f"Train targets: {len(train_targets)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Val targets: {len(val_targets)}")
    
    if args.dset_type == "arrow":
        save_arrow(args, splits, train_samples, train_targets, val_samples, val_targets)
    elif args.dset_type == "h5":
        save_h5(splits, train_samples, train_targets, val_samples, val_targets)
    else:
        raise ValueError("Invalid dset_type")


def save_arrow(args, splits, train_samples, train_targets, val_samples, val_targets):

    binary_t = pa.binary()
    uint16_t = pa.uint16()

    schema = pa.schema([
        pa.field('image_data', binary_t),
        pa.field('label', uint16_t),
    ])

    for split in splits:
        if split == "train":
            samples = train_samples
            targets = train_targets
        else:
            samples = val_samples
            targets = val_targets
        with pa.OSFile(
                os.path.join(args.target_folder, f'ImageNet_{split}.arrow'),
                'wb',
        ) as f:
            with pa.ipc.new_file(f, schema) as writer:
                for (sample, label) in tqdm(zip(samples, targets)):
                    with open(os.path.join(args.data_root, sample), 'rb') as f:
                        img_string = f.read()

                    image_data = pa.array([img_string], type=binary_t)
                    label = pa.array([label], type=uint16_t)

                    batch = pa.record_batch([image_data, label], schema=schema)

                    writer.write(batch)

def save_h5(splits, train_samples, train_targets, val_samples, val_targets):
    with h5py.File(os.path.join(args.target_folder, 'ImageNet.h5'), "w") as g:
        
        for split in splits:
            if split == "train":
                samples = train_samples
                targets = train_targets
            else:
                samples = val_samples
                targets = val_targets
                
            group = g.create_group(split)
            dt_sample = h5py.vlen_dtype(np.dtype(np.uint8))
            dt_target = np.dtype('int16')
            dset = group.create_dataset(
                            'images',
                            (len(samples),),
                            dtype=dt_sample,
                        )
            dtargets = group.create_dataset(
                    'targets',
                    (len(samples),),
                    dtype=dt_target,
                )
            
            for idx, (sample, target) in tqdm(enumerate(zip(samples, targets))):        
                with open(os.path.join(args.data_root, sample), 'rb') as f:

                    img_string = f.read()
                    
                    dset[idx] = np.array(list(img_string), dtype=np.uint8)
                    dtargets[idx] = target
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/p/scratch/training2543/")
    parser.add_argument('--dset_type', choices=['h5', 'arrow'])
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()
    save_files(args)