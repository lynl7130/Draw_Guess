import torch
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import numpy as np


class SketchyDataset(Dataset):
    def __init__(self,
        root_dir
    ):
        super().__init__()
        
        # if root_dir does not exists, quit 
        sketch_dir = os.path.join(root_dir, "sketch")
        assert os.path.exists(root_dir) and os.path.exists(sketch_dir), "Provided Dataset does not exist!"
        
        # read dictionary of classes
        cls_dir = os.path.join(sketch_dir, "tx_000000000000")
        assert os.path.exists(cls_dir), "Cannot read classes because %s does not exist" % cls_dir
        # query for class id by self.classes.index(class name)
        self.classes = sorted(os.listdir(cls_dir)) # sort the classes based on class name
        self.num_classes = len(self.classes)

        self.data = []
        # traverse all sketch samples in sketchy dataset
        # record image path and target class id
        for aug in os.listdir(sketch_dir):
            for cid, cls_ in enumerate(self.classes):
                path = os.path.join(sketch_dir, aug, cls_)
                self.data += [(os.path.join(path, p), cid) for p in os.listdir(path)]
                

    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.array(ImageOps.grayscale(Image.open(self.data[idx][0])))/255.
        tmp = np.empty((img.shape[0], img.shape[1], 2))
        tmp[img == 0, 0] = 1
        tmp[img == 1, 1] = 1

        return {
            "img": tmp, # make sure between 0 and 1
            "tgt": self.data[idx][1]
        }


if __name__ == "__main__":
    dataset = SketchyDataset("/mnt/f/sketchy/256x256")
    print(len(dataset))
    sample = dataset[0]
    print(np.max(sample["img"]), np.min(sample['img']), sample['img'].shape)
    img = Image.fromarray(np.uint8(np.argmax(sample['img'], axis=-1)*255))
    img.save("tmp.png")