import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    # mainly load parameter from yaml file
    parser.add_argument("--exp_yaml", required=True, type=str, help="Config yaml file that describes the exp")
    # able to override yaml if needed
    parser.add_argument("--seed",  required=False, type=int, help="Global seed for the exp")
    parser.add_argument("--lr", required=False, type=float, help="initial learning rate for the exp")
    #parser.add_argument("--input_dim", required=False, type=int, help="input channel size for the model")
    parser.add_argument("--num_classes", required=False, type=int, help="possible sketch classes")
    parser.add_argument("--batch_size", required=False, type=int, help="batch size for training")
    parser.add_argument("--train_base", required=False, type=int, help="how to split train set and val set")
    parser.add_argument("--root_dir", required=False, type=str, help="directory to dataset")
    parser.add_argument("--num_workers", required=False, type=int, help="num_worker for dataloader, exceed 4 -> overheat")
    parser.add_argument("--max_epochs", required=False, type=int, help="train for how many epochs")
    parser.add_argument("--refresh_rate", required=False, type=int, help="visualize loss bar every how many steps")
    parser.add_argument("--log_dir", required=False, type=str, help="directory to store checkpoints and tensorboard logs")
    parser.add_argument("--exp_name", required=False, type=str, help="experiment name")
    parser.add_argument("--model_name", required=False, type=str, help="which model architecture to train/test")
    parser.add_argument("--val_check_interval", required=False, type=float, help="run validation how many times per epoch")
    parser.add_argument("--topk", required=False, type=int, help="accuracy metrics control")
    parser.add_argument("--resume_path", required=False, type=str, help="if want to resume training")
    parser.add_argument("--is_test", action="store_true", help="if passed, only test, do not train")
    parser.add_argument("--flip_bw", required=False, type=bool, help="if False: stroke color is back, background is white; flip if True")
    parser.add_argument("--fps_mode", required=False, type=int, help="if set, load fps sampled sketch alongside with raw sketch")
    args = parser.parse_args()
    
    # open yaml
    with open(args.exp_yaml, 'r') as f:
        config = yaml.safe_load(f)
    # override
    override_dict = {k:v for k,v in vars(args).items() if v is not None and k!="exp_yaml"}
    config.update(override_dict)
    return config

if __name__ == "__main__":
    args = get_args()
    print([key for key in args])