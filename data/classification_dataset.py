import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image, ImageOps
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch_cluster import fps
from torchvision.utils import make_grid
import torchvision
import torch.multiprocessing as mp
import time

class SketchyDataset(Dataset):
    def __init__(self,
        root_dir,
        is_train,
        fps_mode
    ):
        super().__init__()
        
        # if root_dir does not exists, quit 
        sketch_dir = os.path.join(root_dir, "sketch")
        assert os.path.exists(root_dir) and os.path.exists(sketch_dir), "Provided Dataset does not exist!"
        
        
        # read dictionary of classes
        cls_dir = os.path.join(sketch_dir, "tx_000000000000")
        fps_dir = os.path.join(sketch_dir, "fps_000000000000")
        assert os.path.exists(cls_dir), "Cannot read classes because %s does not exist" % cls_dir
        assert os.path.exists(fps_dir), "FPS sampling incomplete!"
        # query for class id by self.classes.index(class name)
        self.classes = sorted(os.listdir(cls_dir)) # sort the classes based on class name
        self.num_classes = len(self.classes)
        self.fps_mode = fps_mode
        self.data = []
        self.fps = []
        # traverse all sketch samples in sketchy dataset
        # record image path and target class id
        for aug in os.listdir(sketch_dir):
            if not aug.startswith("tx_"):
                continue
            for cid, cls_ in enumerate(self.classes):
                path = os.path.join(sketch_dir, aug, cls_)
                fps_path = os.path.join(sketch_dir, aug.replace("tx_", "fps_"), cls_)
                # set aside 1 of 5 sketches for testing
                if is_train:
                    self.data += [(os.path.join(path, p), cid) for p in os.listdir(path) if not p.endswith("5.png")]
                    self.fps += [(os.path.join(fps_path, p), cid) for p in os.listdir(fps_path) if not p.endswith("5.png")]
                else:
                    self.data += [(os.path.join(path, p), cid) for p in os.listdir(path) if p.endswith("5.png")]
                    self.fps += [(os.path.join(fps_path, p), cid) for p in os.listdir(fps_path) if p.endswith("5.png")]
                 

    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.data)

    def load_data(self, path, rgb=False):
        data = Image.open(path)
        data.draft("L", (256, 256))
        if rgb:
            data = ImageOps.grayscale(data)
        return np.asarray(data)/255.

    def __getitem__(self, idx):
        if self.fps_mode==0:
            fps = self.load_data(self.fps[idx][0])
            return {
                "img": np.expand_dims(fps, axis=0), # shape 1x256x256
                "tgt": self.data[idx][1]
            }
        elif self.fps_mode == 4:
            fps = self.load_data(self.fps[idx][0])
            fps = np.eye(2)[fps.reshape(-1).astype(int)].reshape(fps.shape[0], fps.shape[1], 2).transpose(2, 0, 1)
            return {
                "img": fps, # shape 2x256x256
                "tgt": self.data[idx][1]
            }
        elif self.fps_mode==1: 
            img = self.load_data(self.data[idx][0], True)
            fps = self.load_data(self.fps[idx][0])
            result = None
            try: 
                result = np.concatenate([np.expand_dims(img, axis=0), np.expand_dims(fps, axis=0)], axis=0) # 2x256x256
            except:
                print(img.shape, fps.shape)
            return {
                "img": result,
                "tgt": self.data[idx][1]
            }
        elif self.fps_mode==2:
            img = self.load_data(self.data[idx][0], True)
            return {
                "img": np.expand_dims(img, axis=0), # shape 1x256x256
                "tgt": self.data[idx][1]
            }
        elif self.fps_mode == 3:
            img = self.load_data(self.data[idx][0], True)
            fps = self.load_data(self.fps[idx][0])
            fps = np.eye(2)[fps.reshape(-1).astype(int)].reshape(fps.shape[0], fps.shape[1], 2).transpose(2, 0, 1)
            result = None
            try: 
                result = np.concatenate([np.expand_dims(img, axis=0), fps], axis=0) # 3x256x256
            except:
                print(img.shape, fps.shape)
            return {
                "img": result,
                "tgt": self.data[idx][1]
            }
        else:
            assert False, "Invalid fps mode!"

class SketchyDataModule(pl.LightningDataModule):
    def __init__(self, 
        root_dir,
        fps_mode,
        train_ratio=11/12.,
        batch_size=32,
        num_workers=8):
        super().__init__()
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.fps_mode = fps_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None
        self.data_test = None
    def calc_steps_per_epoch(self):
        if self.data_train:
            return len(self.data_train) // self.batch_size
        else:
            data_full = SketchyDataset(self.root_dir, is_train=True, fps_mode=self.fps_mode)
            num_train = int(self.train_ratio * len(data_full))
            return num_train // self.batch_size

    def setup(self, stage= None):
        if stage == "fit" or stage is None:
            data_full = SketchyDataset(self.root_dir, is_train=True, fps_mode=self.fps_mode)
            num_train = int(self.train_ratio * len(data_full))
            self.data_train, self.data_val = random_split(data_full, [num_train, len(data_full) - num_train])
        else:
            self.data_test = SketchyDataset(self.root_dir, is_train=False, fps_mode=self.fps_mode)
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)

def worker(aug, sketch_dir, cls_dir, classes):
    pbar = tqdm(total=len(classes))
    for cid, cls_ in enumerate(classes):
        path = os.path.join(sketch_dir, aug, cls_)
        fps_path = os.path.join(sketch_dir, aug.replace("tx_", "fps_"), cls_)
        try:    
            os.makedirs(fps_path)
        except:
            pass
        for p in os.listdir(path):
            img = np.array(ImageOps.grayscale(Image.open(os.path.join(path, p))))/255.
            img = torch.from_numpy(img).to('cuda')
            save_dir = os.path.join(fps_path, p)
            if os.path.exists(save_dir):
                continue
            # fps sampling from img
            # img; Bx3xHxW
            # prepare sampling candidate xy: BxHxWx2
            xy = torch.stack(torch.meshgrid(
                torch.linspace(0, img.shape[-2]-1, img.shape[-2]),
                torch.linspace(0, img.shape[-1]-1, img.shape[-1]),
                indexing='ij'
            ), dim=-1).to(img.device) 
            #print(xy[10, 98])
            xy = xy.view(-1, 2)
            candid = (img < 0.5).view(-1)
            xy = xy[candid, :]
            batch = torch.zeros((xy.shape[0],), device=xy.device).long()
            #xy = xy.view(1, img.shape[-2], img.shape[-1], 2).repeat(img.shape[0], 1, 1, 1).view(-1, 2)
            # B images, each has black white -> 2*B batches in total
            #batch = 2*torch.linspace(0, img.shape[0]-1, img.shape[0]).to(img.device) # (B,)
            #batch = batch.unsqueeze(-1).repeat(1, img.shape[2]*img.shape[-1]).flatten().view(img.shape[0], img.shape[-2], img.shape[-1])
            #batch[img[:,0,...]>0.5] += 1
            #batch = batch.view(-1)
            #assert False, torch.unique(img)
            #print(batch[2*256*256 + 5*256+ 76])
            #print(batch[20*256*256+ 232*256+ 255])
            #assert False, (xy.dtype, batch.dtype, xy.shape, batch.shape)
            index = fps(xy, batch.long(), ratio=0.25, random_start=False).long() 
            xy = xy.long()
            new = torch.ones_like(img)
            new[xy[index, 0], xy[index, 1]] = 0
            #grid = torch.cat([img, new], dim=-1)
            #grid = make_grid([img, new])
            #assert False, ("size of grid:", grid.size())
            tmp = torchvision.transforms.ToPILImage()(new)
            
            #assert False, (os.path.join(path, p), save_dir)
            tmp.save(save_dir)
            #batch = torch.ones((xy.shape[0], xy.shape[1], xy.shape[2]), device=img.device)
            
            #assert False, xy.shape

            #assert False, img.shape
        pbar.update(1)
    pbar.close()

def fps_sampling(sketch_dir):
    cls_dir = os.path.join(sketch_dir, "tx_000000000000")
    assert os.path.exists(cls_dir), "Cannot read classes because %s does not exist" % cls_dir
    classes = sorted(os.listdir(cls_dir)) 
    
    processes = []
    for aug in os.listdir(sketch_dir):
        if not aug.startswith("tx_"):
            continue
        print("processing %s..." % aug)
        p = mp.Process(target=worker, args=(aug,sketch_dir, cls_dir, classes))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

        


if __name__ == "__main__":
    #dataset = SketchyDataset("/mnt/f/sketchy/256x256", True)
    #print(len(dataset))
    #sample = dataset[0]
    #print(np.max(sample["img"]), np.min(sample['img']), sample['img'].shape)
    #img = Image.fromarray(np.uint8(sample['img'][0]*255))
    #img.save("tmp.png")
    ''' 
    # fps generation 
    mp.set_start_method('spawn')
    fps_sampling("/mnt/f/sketchy/256x256/sketch")
    '''

    # speed up image loading
    path = "/mnt/f/sketchy/256x256/sketch/fps_000000000000/bear/n02131653_808-5.png"
    start = time.time()
    fps = np.array(ImageOps.grayscale(Image.open(path)))/255.  
    end = time.time()
    print("old one", end - start)
    start = time.time()
    fps = Image.open(path)
    fps.draft("L", (256, 256))
    fps = np.asarray(fps)/255.
    fps = np.eye(2)[fps.reshape(-1).astype(int)].reshape(fps.shape[0], fps.shape[1], 2).transpose(2, 0, 1)
    assert False, (fps.shape, fps)
    end = time.time()
    print("new one: ", end - start)
    start = time.time()
    fps = Image.open(path)
    fps.draft("L", (256, 256))
    fps = ImageOps.grayscale(fps)
    fps = np.asarray(fps)/255.
    end = time.time()
    print("rgb one: ", end - start)
    Image.fromarray((fps*255.).astype(np.uint8)).save("tmp.png")