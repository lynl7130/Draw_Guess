import runners
import data
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from configs import get_args
import os

config = get_args()
train_ratio = float(config["train_base"]-1) / float(config["train_base"])


seed_everything(config["seed"])

datamodule = data.SketchyDataModule(
    root_dir=config["root_dir"],
    train_ratio=train_ratio,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    )


steps_per_epoch = datamodule.calc_steps_per_epoch()

model = runners.Classifier(lr=config["lr"],
    model_name=config["model_name"],
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    batch_size=config["batch_size"],
    steps_per_epoch = steps_per_epoch,
    topk = config["topk"],
    flip_bw=config["flip_bw"],
    config=config)
model.datamodule = datamodule

# tensorboard logger will automatically create a version directory for us under log_dir/exp_name 
# need do this ahead because ModelCheckpoint rely on logger to decide checkpoing save dir
logger = TensorBoardLogger(config["log_dir"], name=config["exp_name"])
save_ckpt = os.path.join(logger.log_dir, "checkpoints")
assert not os.path.exists(save_ckpt), "checkpoint dir none empty!"
os.makedirs(save_ckpt)

trainer = Trainer(
    max_epochs = config["max_epochs"],
    gpus=min(1, torch.cuda.device_count()),
    logger=logger,
    callbacks=[LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=config["refresh_rate"]),
        ModelCheckpoint(
            dirpath=save_ckpt,
            filename="{epoch}-{step}-{val_loss:.2f}",
            monitor="val_loss",
            save_last=True,
            save_top_k=2,
            mode="min"
            )
    ],
    val_check_interval=config["val_check_interval"]
)
if not config['is_test']:
    # even when starting from an existing checkpoint path, will create a new version directory
    trainer.fit(model, model.datamodule, ckpt_path=config["resume_path"] if config["resume_path"] != "" else None)
    trainer.test(model, model.datamodule)
else:
    # note, even when doing merely testing, will be a seperate dir created for this run
    assert config["resume_path"] is not None, "Test ckpt not provided!"
    assert config["resume_path"].endswith(".ckpt") and os.path.exists(config["resume_path"]), "Not a legal test ckpt!"
    trainer.test(model, model.datamodule, ckpt_path=config["resume_path"])
    