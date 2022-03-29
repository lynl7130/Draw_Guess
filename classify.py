import runners
import data
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


seed_everything(7)

model = runners.Classifier(lr=0.05)

model.datamodule = data.SketchyDataModule(
    root_dir="/mnt/f/sketchy/256x256"
)

trainer = Trainer(
    max_epochs = 1,
    gpus=min(1, torch.cuda.device_count()),
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=30)
    ],

)

trainer.fit(model, model.datamodule)
trainer.test(model, model.datamodule)