
from .ResNet import create_model as create_ResNet


def create_model(config):
    if config["model_name"]=="ResNet":
        model = create_ResNet(
            config["input_dim"],
            config["num_classes"]
        )
    else:
        assert False, "Undefined model name %s" % config["model_name"]
    return model

#from .ResNet import create_model as create_ResNet