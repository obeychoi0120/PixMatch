import importlib
import torch


def get_model(args):

    model = getattr(importlib.import_module(args.network), 'Net')(num_classes=args.num_classes)

    if args.weights[-7:] == '.params':
        assert args.network in ["network.resnet38_cls",
                                "network.resnet38_seam",
                                "network.resnet38_eps",
                                "network.resnet38_eps_seam",
                                "network.resnet38_contrast",
                                "network_with_PCM.resnet38_eps_seam_p",
                                "network.resnet50_cam"
                                ]
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    return model
