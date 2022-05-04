
from .ResNet import create_model as create_ResNet
from .cnn import create_model as create_cnn
from .vit import create_model as create_vit
from .m_vit import create_model as create_mvit

def create_model(config):
    if config["model_name"]=="ResNet":
        model = create_ResNet(
            config["input_dim"],
            config["num_classes"]
        )
    elif config["model_name"]=="CNN":
        model = create_cnn(
            config["input_dim"],
            config["num_classes"]
        )
    elif config["model_name"]=="VIT":
        model = create_vit(
            config["num_classes"]
        )
    elif config["model_name"]=="MVIT":
        model = create_mvit(
            config["num_classes"]
        )


    else:
        assert False, "Undefined model name %s" % config["model_name"]
    return model

#from .ResNet import create_model as create_ResNet