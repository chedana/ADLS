import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current script directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
import argparse
from acnet.acnet_builder import ACNetBuilder
from ndp_train import *
from acnet.acnet_fusion import convert_acnet_weights
import os
from ndp_test import general_test
from constants import LRSchedule
from builder import ConvBuilder
import shutil

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config('config.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', default='None')
    # Example usage
    # import pdb;pdb.set_trace()



    # parser.add_argument('-a', '--arch', default='sres18')
    # parser.add_argument('-b', '--block_type', default='acb')
    parser.add_argument('-c', '--conti_or_fs', default='fs')        # continue or train_from_scratch
    parser.add_argument('-e', '--eval',default= False)
    # parser.add_argument('-kd', '--teacher_net',default= None)
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')

    start_arg = parser.parse_args()
    config_path = start_arg.config

    config = load_config(config_path)

    network_type = config['model']['name']
    block_type = config['model']['block_type']
    conti_or_fs =start_arg .conti_or_fs
    eval_mode = start_arg.eval
    if eval_mode:
        ckpt_path = config['model']['ckpt']
    
    KD = config['teacher']['KD']
    if KD:
        teacher_config = config['teacher']
        # teacher_config['teacher_net'] = config['teacher']['teacher_net']
        # teacher_config['ckpt'] = config['teacher']['ckpt']
        # teacher_config['block_type'] = config['teacher']['block_type']
    else:
        teacher_config = None


    batch_size = int(config['training']['batch_size'])
    epochs = int(config['training']['epochs'])
    assert conti_or_fs in ['continue', 'fs']
    assert block_type in ['acb', 'base']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)

    gamma_init = None

    if network_type == 'sres18':
        weight_decay_strength = 1e-4
        batch_size = 256
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 1

    elif network_type == 'sres34':
        weight_decay_strength = 1e-4
        batch_size = 256
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 1

    elif network_type == 'sres50':
        weight_decay_strength = 1e-4
        batch_size = 256
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 1

    elif network_type == 'lenet5bn':
        weight_decay_strength = 1e-4

        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
    elif network_type == 'lenet5bn_deep':
        weight_decay_strength = 1e-4

        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)

    elif network_type == 'mobilev1cifar':
        weight_decay_strength = 1e-4

        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
    elif network_type == 'mobilev1cifar_shallow':
        weight_decay_strength = 1e-4

        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
    elif network_type == 'cfqkbnc':
        weight_decay_strength = 1e-4
        #   ------------------------------------
        #   86.2  --->  86.8+
        #50 epoch 84.69  --->  85.75+
        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
    
    elif network_type == 'cfqkbnc_deep':
        weight_decay_strength = 1e-4
        #   ------------------------------------
        #50 epoch 87.97  --->  88.51+
        batch_size = batch_size
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   ------------------------------------
        #   ------------------------------------
    elif network_type == 'eca_cfqkbnc':
        weight_decay_strength = 1e-4
        #   ------------------------------------
        #   86.2  --->  86.8+
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=150, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   ------------------------------------

    elif network_type == 'src56':
        weight_decay_strength = 1e-4
        #   ------------------------------------
        #   94.47  --->  95+
        batch_size = 128
        lrs = LRSchedule(base_lr=0.2, max_epochs=400, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   --------------------------------------

    elif network_type == 'vc':
        weight_decay_strength = 1e-4
        #   --------------------------------------
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   --------------------------------------


    elif network_type == 'vc_shallow':
        weight_decay_strength = 1e-4
        #   --------------------------------------
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   --------------------------------------

    elif network_type == 'wrnc16plain':
        weight_decay_strength = 5e-4
        #   --------------------------------------
        #   95.90 -> 96.33
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=400, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        gamma_init = 0.333
        #   --------------------------------------
    else:
        raise ValueError('...')
    # import pdb;pdb.set_trace()
    from datetime import date
    today = date.today()
    if KD:
        teacher_info = '_'.join([config['teacher']['teacher_net'],config['teacher']['block_type'],config['teacher']['method']])
        log_dir = str(today)+'/{}_{}_epoch_{}_kd_{}'.format(network_type, block_type,epochs,teacher_info)
    else:
        log_dir = str(today)+'/{}_{}_epoch_{}_train'.format(network_type, block_type,epochs)

    weight_decay_bias = weight_decay_strength
    config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries, cosine_minimum=lrs.cosine_minimum,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=5, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=None)

    if block_type == 'acb':
        builder = ACNetBuilder(base_config=config, deploy=False, gamma_init=gamma_init)
    else:
        builder = ConvBuilder(base_config=config)
    if eval_mode:
        target_weights = os.path.join(ckpt_path, 'finish.hdf5')
    else:
        target_weights = os.path.join(log_dir, 'finish.hdf5')
    # import pdb;pdb.set_trace()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        shutil.copy(config_path, log_dir)
    else:
        shutil.copy(config_path, log_dir)
    if not os.path.exists(target_weights):
        if KD:
            if teacher_config['block_type'] == 'acb':
                teacher_builder = ACNetBuilder(base_config=config, deploy=False, gamma_init=gamma_init)
            else:
                teacher_builder = ConvBuilder(base_config=config)
            teacher_config['teacher_builder'] = teacher_builder
            train_kd_main(local_rank=start_arg.local_rank,teacher_config = teacher_config, cfg=config, convbuilder=builder,
                   show_variables=True, auto_continue=auto_continue)
        else:
            train_main(local_rank=start_arg.local_rank, cfg=config, convbuilder=builder,
                   show_variables=True, auto_continue=auto_continue)

    if block_type == 'acb' and start_arg.local_rank == 0:
        convert_acnet_weights(target_weights, target_weights.replace('.hdf5', '_deploy.hdf5'), eps=1e-5)
        deploy_builder = ACNetBuilder(base_config=config, deploy=True)
        general_test(network_type=network_type, weights=target_weights.replace('.hdf5', '_deploy.hdf5'),
                 builder=deploy_builder)
    else:
        # convert_acnet_weights(target_weights, target_weights.replace('.hdf5', '_deploy.hdf5'), eps=1e-5)
        deploy_builder = ConvBuilder(base_config=config)
        general_test(network_type=network_type, weights=target_weights,
                 builder=deploy_builder)
