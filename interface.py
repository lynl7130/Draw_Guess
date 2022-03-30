import gradio as gr
import numpy as np
from models import create_ResNet
import torch
from PIL import Image

device = torch.device("cuda") 
#model = create_ResNet(
#                2,
#                250
#            ).to(device)

def sketch_recognition(img):
    
    # img will be fed in as numpy array with shape (28, 28)
    
    # note: this input need wash! 
    # while our input size is (2, 256, 256)
    # linear input (0-255) while ours discrete: 0/1 
    print(np.unique(img), np.max(img), np.min(img))
    
    return ",".join([str(s) for s in img.shape])
    

#iface = gr.Interface(fn=sketch_recognition, inputs="sketchpad", outputs="label").launch()
iface = gr.Interface(fn=sketch_recognition, inputs="sketchpad", outputs="text").launch()

iface.launch()