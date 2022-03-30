import runners
import data
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from configs import get_args

config = get_args()
train_ratio = float(config["train_base"]-1) / float(config["train_base"])


seed_everything(config["seed"])

datamodule = data.SketchyDataModule(
    root_dir=config["root_dir"],
    train_ratio=train_ratio,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"])


steps_per_epoch = datamodule.calc_steps_per_epoch()

model = runners.Classifier(lr=config["lr"],
    model_name=config["model_name"],
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    batch_size=config["batch_size"],
    steps_per_epoch = steps_per_epoch)
model.datamodule = datamodule


trainer = Trainer(
    max_epochs = config["max_epochs"],
    gpus=min(1, torch.cuda.device_count()),
    logger=TensorBoardLogger(config["log_dir"], name=config["exp_name"]),
    callbacks=[LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=config["refresh_rate"])
    ],
    val_check_interval=config["val_check_interval"]
)
trainer.fit(model, model.datamodule)
trainer.test(model, model.datamodule)