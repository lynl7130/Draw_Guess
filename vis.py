import gradio as gr
import numpy as np
from models import create_model
import torch
from PIL import Image
from configs import get_args
import os
from scipy import ndimage
import torch.nn.functional as F
from data import SketchyDataset
from torch_cluster import fps
import torchvision

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


threshold=250

config = get_args()
print("Loading model architecture...")
model = create_model(config).to(device)

print("Loading checkpoint...")
assert config["resume_path"] is not None, "Vis ckpt not provided!"
assert config["resume_path"].endswith(".ckpt") and os.path.exists(config["resume_path"]), "Not a legal vis ckpt!"
ckpt = torch.load(config["resume_path"])["state_dict"]
for name in model.state_dict():
    model.state_dict()[name].copy_(ckpt["model." + name])
model.eval()

print("Loading class map...")
data_full = SketchyDataset(config["root_dir"], is_train=True, fps_mode=config["fps_mode"])
classes = data_full.get_class()  


def sketch_recognition(img):
    
    # input preprocessing
    # value falls in between 0 - 255, not 0/255 as in our case
    # also, the brush is thicker than ours, so thresholding to be 0/255
    # followed by dilation
    '''
    tmp = np.zeros_like(img).astype(float)
    tmp[img > threshold] = 1.
    img = ndimage.binary_dilation(tmp, structure=np.ones((10,10)).astype(tmp.dtype)).astype(tmp.dtype)
    
    Image.fromarray(np.uint8(img * 255)).save("test.png")
    '''
    # form input  
    # time-consuming part! 10s
    x = torch.from_numpy(img/255.).to(device).float()   
    #print(x.shape, torch.max(x), torch.min(x))
    #return ",".join([x.shape, torch.max(x), torch.min(x)])

    
    # extract fps for this sketch
    #img = torch.from_numpy(x).to('cuda')
    #save_dir = os.path.join(fps_path, p)
    #if os.path.exists(save_dir):
    #    continue
    # fps sampling from img
    # img; Bx3xHxW
    # prepare sampling candidate xy: BxHxWx2
    xy = torch.stack(torch.meshgrid(
        torch.linspace(0, x.shape[-2]-1, x.shape[-2]),
        torch.linspace(0, x.shape[-1]-1, x.shape[-1]),
        indexing='ij'
    ), dim=-1).to(x.device) 
    #print(xy[10, 98])
    xy = xy.view(-1, 2)
    candid = (x < 0.5).view(-1)
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
    new = torch.ones_like(x)
    new[xy[index, 0], xy[index, 1]] = 0
    #tmp = torchvision.transforms.ToPILImage()(new)
    #assert False, (os.path.join(path, p), save_dir)
    #tmp.save("tmp.png")
    new = torch.nn.functional.one_hot(new.long(), 2).permute(2, 0, 1).unsqueeze(0)
    if config["fps_mode"] == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif config["fps_mode"] == 3:
        x = torch.cat((x.unsqueeze(0).unsqueeze(0), new), dim=1)
    else:
        assert False, "not supported for now"
    #grid = torch.cat([img, new], dim=-1)
    #grid = make_grid([img, new])
    #assert False, ("size of grid:", grid.size())
    #tmp = torchvision.transforms.ToPILImage()(new)
    
    # pass model to get logits
    if config["flip_bw"]:
        x = 1. - x
    out = model(x)
    logits = F.softmax(out, dim=1)

    topk = torch.topk(logits, k=config["topk"], dim=-1)
    values = topk.values
    indices = topk.indices
    results = {}
    for i, t in enumerate(indices.flatten()):
        results[classes[int(t)]] = float(values[0][i])
    return results
    
    
if __name__ == "__main__":


    #iface = gr.Interface(fn=sketch_recognition, inputs="sketchpad", outputs="label").launch()
    iface = gr.Interface(fn=sketch_recognition, 
        inputs=gr.inputs.Image(
            shape=(256, 256), 
            image_mode="L", 
            invert_colors=False, 
            source="upload",
            #source="canvas"
        ), 

        outputs="label").launch()

    iface.launch()