import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from models import create_ResNet
import torch
from torchmetrics.functional import accuracy

class Classifier(pl.LightningModule):
    def __init__(self, 
        lr=0.05,
        model_name="ResNet",
        input_dim=2,
        num_classes=250,
        ):
        super().__init__()
        self.batch_size = 32
        self.steps_per_epoch = 11808
        print("note, this is a dead batch size / steps_per_epoch for now")
        self.save_hyperparameters()
        if model_name=="ResNet":
            self.model = create_ResNet(
                input_dim,
                num_classes
            )
        else:
            assert False, "Undefined model name %s" % model_name
    def forward(self, x):
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
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

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
        
