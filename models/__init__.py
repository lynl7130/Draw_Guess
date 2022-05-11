
from .ResNet import create_model as create_ResNet
from .cnn import create_cnn, create_AS
from .cnn_vit import create_model as create_vit
from .m_vit import create_model as create_mvit

def create_model(config):
    if config["fps_mode"] in [0, 2]:
        input_dim = 1
    elif config["fps_mode"] == 1:
        input_dim = 2
    else:
        assert False, "invalid fps mode"
    if config["model_name"]=="ResNet":
        model = create_ResNet(
            input_dim,
            config["num_classes"]
        )
    elif config["model_name"]=="CNN":
        model = create_cnn(
            input_dim,
            config["num_classes"]
        )
    elif config["model_name"]=="VIT":
        model = create_vit(
            input_dim,
            config["num_classes"]
        )
    elif config["model_name"]=="MVIT":
        model = create_mvit(
            input_dim,
            config["num_classes"]
        )
    elif config["model_name"]=="AS":
        assert input_dim == 2, "ASNet requires two channels"
        model = create_AS(
            input_dim,
            config["num_classes"]
        )

    else:
        assert False, "Undefined model name %s" % config["model_name"]
    return model

#from .ResNet import create_model as create_ResNet