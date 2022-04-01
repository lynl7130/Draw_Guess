import gradio as gr
import numpy as np
from models import create_ResNet
import torch
from PIL import Image
from configs import get_args
import os
from scipy import ndimage
import torch.nn.functional as F
from data import SketchyDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

threshold=250

config = get_args()
print("Loading model architecture...")
if config["model_name"]=="ResNet":
    model = create_ResNet(
        config["input_dim"],
        config["num_classes"]
        ).to(device)
else:
    assert False, "Undefined model name %s" % config["model_name"]

print("Loading checkpoint...")
assert config["resume_path"] is not None, "Vis ckpt not provided!"
assert config["resume_path"].endswith(".ckpt") and os.path.exists(config["resume_path"]), "Not a legal vis ckpt!"
ckpt = torch.load(config["resume_path"])["state_dict"]
for name in model.state_dict():
    model.state_dict()[name].copy_(ckpt["model." + name])
model.eval()

print("Loading class map...")
data_full = SketchyDataset(config["root_dir"], is_train=True)
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
    x = torch.from_numpy(img/255.).to(device).view(1, 1, img.shape[0], img.shape[1]).float()   
    #print(x.shape, torch.max(x), torch.min(x))
    #return ",".join([x.shape, torch.max(x), torch.min(x)])

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