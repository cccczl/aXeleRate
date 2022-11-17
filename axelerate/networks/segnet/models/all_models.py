from . import pspnet
from . import unet
from . import segnet
from . import fcn
model_from_name = {
    "fcn_8": fcn.fcn_8,
    "fcn_32": fcn.fcn_32,
    "fcn_8_vgg": fcn.fcn_8_vgg,
    "fcn_32_vgg": fcn.fcn_32_vgg,
    "fcn_8_resnet50": fcn.fcn_8_resnet50,
    "fcn_32_resnet50": fcn.fcn_32_resnet50,
    "fcn_8_mobilenet": fcn.fcn_8_mobilenet,
    "fcn_32_mobilenet": fcn.fcn_32_mobilenet,
    "pspnet": pspnet.pspnet,
    "vgg_pspnet": pspnet.vgg_pspnet,
    "resnet50_pspnet": pspnet.resnet50_pspnet,
    "pspnet_50": pspnet.pspnet_50,
    "pspnet_101": pspnet.pspnet_101,
    "unet_mini": unet.unet_mini,
    "unet": unet.unet,
    "vgg_unet": unet.vgg_unet,
    "resnet50_unet": unet.resnet50_unet,
    "mobilenet_unet": unet.mobilenet_unet,
    "segnet": segnet.segnet,
    "vgg_segnet": segnet.vgg_segnet,
    "resnet50_segnet": segnet.resnet50_segnet,
    "mobilenet_segnet": segnet.mobilenet_segnet,
}
