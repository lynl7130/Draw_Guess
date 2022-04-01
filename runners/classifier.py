import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from models import create_model
import torch
from torchmetrics.functional import accuracy
from PIL import Image
import numpy as np

class Classifier(pl.LightningModule):
    def __init__(self, 
        lr=0.05,
        model_name="ResNet",
        input_dim=1,
        num_classes=250,
        batch_size = 32,
        steps_per_epoch = None,
        topk = 5,
        flip_bw=False,
        config=None
        ):
        super().__init__()
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch 
        self.topk = topk
        self.flip_bw = flip_bw
        self.save_hyperparameters()
        self.model = create_model(
            config
        )
    def forward(self, x):
        if self.flip_bw:
            x = 1. - x
        out = self.model(x)
        return F.log_softmax(out, dim=1)
    def training_step(self, batch, batch_idx):
        x = batch["img"].float()
        y = batch["tgt"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss
    def evaluate(self, batch, stage=None):
        x = batch["img"].float()
        y = batch["tgt"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            for k in range(1, self.topk+1):
                self.log(f"{stage}_acc@%s" % k, accuracy(logits, y, top_k=k), prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        
        #steps_per_epoch = 45000 // self.batch_size
        #assert False, "should change the 45000 above! not cifar anymore"
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
            ),
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
