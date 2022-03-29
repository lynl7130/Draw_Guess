import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image, ImageOps
import numpy as np
import pytorch_lightning as pl


class SketchyDataset(Dataset):
    def __init__(self,
        root_dir,
        is_train
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
                # set aside 1 of 5 sketches for testing
                if is_train:
                    self.data += [(os.path.join(path, p), cid) for p in os.listdir(path) if not p.endswith("5.png")]
                else:
                    self.data += [(os.path.join(path, p), cid) for p in os.listdir(path) if p.endswith("5.png")]
                 

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
            "img": tmp.transpose(2, 0, 1), # shape 2x256x256, classify as has stroke/no stroke
            "tgt": self.data[idx][1]
        }

class SketchyDataModule(pl.LightningDataModule):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
    def setup(self, stage= None):
        if stage == "fit" or stage is None:
            data_full = SketchyDataset(self.root_dir, is_train=True)
            num_train = int(11/12. * len(data_full))
            self.data_train, self.data_val = random_split(data_full, [num_train, len(data_full) - num_train])
        else:
            self.data_test = SketchyDataset(self.root_dir, is_train=False)
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=32, num_workers=16)
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=32, num_workers=16)
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=32, num_workers=16)



if __name__ == "__main__":
    dataset = SketchyDataset("/mnt/f/sketchy/256x256", True)
    print(len(dataset))
    sample = dataset[0]
    print(np.max(sample["img"]), np.min(sample['img']), sample['img'].shape)
    img = Image.fromarray(np.uint8(np.argmax(sample['img'], axis=0)*255))
    img.save("tmp.png")