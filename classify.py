import runners
import data
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


seed = 7
lr = 0.05
input_dim = 1
num_classes = 250
batch_size = 32
train_ratio = 11 / 12.
root_dir = "/mnt/f/sketchy/256x256"
num_workers = 4 #exceed -> overheat
max_epochs = 1
refresh_rate = 30
log_dir = "lightning_logs/"
exp_name = "resnet"
model_name = "ResNet"

seed_everything(seed)

datamodule = data.SketchyDataModule(
    root_dir=root_dir,
    train_ratio=train_ratio,
    batch_size=batch_size,
    num_workers=num_workers)


steps_per_epoch = datamodule.calc_steps_per_epoch()

model = runners.Classifier(lr=lr,
    model_name=model_name,
    input_dim=input_dim,
    num_classes=num_classes,
    batch_size=batch_size,
    steps_per_epoch = steps_per_epoch)
model.datamodule = datamodule


trainer = Trainer(
    max_epochs = max_epochs,
    gpus=min(1, torch.cuda.device_count()),
    logger=TensorBoardLogger(log_dir, name=exp_name),
    callbacks=[LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=refresh_rate)
    ],

)

trainer.fit(model, model.datamodule)
trainer.test(model, model.datamodule)