import torch
import torch.nn as nn
import sys
import os
import datetime
import argparse
import numpy as np

import utils
import loaders
import models
import supervision
import metrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation"
        "Usage:  python test.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #model details
    parser.add_argument('-bs','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='sdnet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('--load_weights_path', type=str, default=None, help= 'Path to save model checkpoints')
    #data
    parser.add_argument('--data_path', type= str, default='/home/sthermos/idcom_imaging/data/Cardiac/ACDC/segmentation/training', help = 'Path to input data')
    parser.add_argument('--save_path', type=str, default='factors', help='Path to save the anatomy factors')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='-1', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')
    #visdom params
    parser.add_argument('-d','--disp_iters', type=int, default=10, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom', type=str, nargs='?', default=None, const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument('--visdom_iters', type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    #create and init device
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))    
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Testing {0} on {1}'.format(args.name, device))

    #Visdom setup
    visualizer = utils.visualization.NullVisualizer() if args.visdom is None\
        else utils.visualization.VisdomVisualizer(args.name, args.visdom, count=1)
    if args.visdom is None:
        args.visdom_iters = 0

    loaded_params = torch.load(args.load_weights_path, map_location=device) #, map_location=device
    #Model selection and initialization
    model_params = {
        'width': loaded_params['width'],
        'height': loaded_params['height'],
        'ndf': loaded_params['ndf'],
        'norm': loaded_params['norm'],
        'upsample': loaded_params['upsample'],
        'num_classes': loaded_params['num_classes'],
        'decoder_type': loaded_params['decoder_type'],
        'anatomy_out_channels': loaded_params['anatomy_out_channels'],
        'z_length': loaded_params['z_length'],
        'num_mask_channels': loaded_params['num_mask_channels']
    }
    model = models.get_model(args.model_name, model_params)
    model.load_state_dict(loaded_params['model_state_dict'])
    model.to(device)
    model.eval()
    
    #load test set
    loader = loaders.ACDCLoader(args.data_path)
    tdata = loader.load_labelled_data(0, 'test')
    print(tdata.images.shape, tdata.masks.shape)
    print(tdata.volumes(), len(tdata.volumes()))
    timages = torch.from_numpy(tdata.images.astype(float))
    tmasks = torch.from_numpy(tdata.masks.astype(float))

    #Dice score variable initialization
    test_running_dice_score = supervision.AverageMeter()
    test_running_std = supervision.AverageMeter()

    #auxiliary tensors init
    t_image = torch.zeros(1, 1, 224, 224)
    t_mask = torch.zeros(1, 4, 224, 224)

    with torch.no_grad():
        for iteration in range(timages.shape[0]):
            t_image[0] = timages[iteration]
            #add an extra channel for mask background
            cmask = tmasks[iteration]
            logical_or = torch.sum(cmask, dim=0)
            tmpmask_0 = 1 - logical_or
            tmpmask = torch.cat([tmpmask_0.unsqueeze(0), cmask], dim=0)
            t_mask[0] = tmpmask

            #forward pass
            _, _, _, _, a_out, seg_pred, _, _, _, _ = model(t_image.to(device), t_mask.to(device), 'test')
            #dice score computation
            dice_score = supervision.dice_score(seg_pred[:,1:,:,:], t_mask[:,1:,:,:].to(device))
            #logging
            test_running_dice_score.update(dice_score)
            utils.save_anatomy_factors(a_out[0].cpu().numpy(), args.save_path, iteration)


        print("Test Samples: {}\nDice: {}\n"\
            .format(iteration+1, test_running_dice_score.avg))

        test_running_dice_score.reset()
