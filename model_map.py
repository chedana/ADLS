from base_model.mobilenetv1 import *
from base_model.stagewise_resnet import *
from base_model.vgg import *
from base_model.lenet5 import *
from base_model.wrn import create_wrnc16plain
from base_model.resnet import create_ResNet18, create_ResNet34
from base_model.cfqk import *

IMAGENET_STANDARD_MODEL_MAP = {
    'smi1': create_MobileV1Imagenet,
    # 'sres18': create_ResNet18,
    # 'sres34': create_ResNet34
}

CIFAR10_MODEL_MAP = {
    'src56':create_SRC56,
    'src110':create_SRC110,
    'vc':create_vc,
    'wrnc16plain':create_wrnc16plain,
    'cfqkbnc':create_CFQKBNC,
    'eca_cfqkbnc':create_eca_CFQKBNC,
    'cfqkbnc_deep':create_CFQKBNC_deep,
    'lenet5bn': create_lenet5bn,
    'lenet5bn_deep': create_lenet5bn_deep,
    'mobilev1cifar': create_MobileV1Cifar
    'mobilev1cifar_shallow': MobileV1CifarNet_shallow
}


CIFAR10_res_MODEL_MAP = {
    'sres18': create_ResNet18,
    'sres34': create_ResNet34,
    'sres50': create_SResNet50,
}
MNIST_MODEL_MAP = {
    # 'lenet5bn': create_lenet5bn,
    # 'lenet5bn_deep': create_lenet5bn_deep,
}

DATASET_TO_MODEL_MAP = {
    'imagenet_standard': IMAGENET_STANDARD_MODEL_MAP,
    'cifar10': CIFAR10_MODEL_MAP,
    'cifar10_res': CIFAR10_res_MODEL_MAP,
    'mnist': MNIST_MODEL_MAP
}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    # print(DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')].keys())
    return DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')][model_name]

def get_dataset_name_by_model_name(model_name):
    for dataset_name, model_map in DATASET_TO_MODEL_MAP.items():
        if model_name in model_map:
            return dataset_name
    return None