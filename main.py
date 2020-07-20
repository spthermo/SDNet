import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import datetime
import argparse

import utils
import loaders
import models
import supervision

#hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=10, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='sdnet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument("--anatomy_factors", type=int, default=8, help = 'Number of anatomy factors to encode')
    parser.add_argument("--modality_factors", type=int, default=8, help = 'Number of modality factors to encode')
    parser.add_argument("--charbonnier", type=int, default=0, help = 'Choose Charbonnier penalty for the reconstruction loss')
    parser.add_argument("--data_path", type=str, default='data', help='Path to ACDC dataset')
    #regularizers weights
    parser.add_argument("--kl_w", type=float, default=0.01, help = 'KL divergence loss weight')
    parser.add_argument("--regress_w", type=float, default=1.0, help = 'Regression loss weight')
    parser.add_argument("--focal_w", type=float, default=0.0, help = 'Focal loss weight')
    parser.add_argument("--dice_w", type=float, default=10.0, help = 'Dice loss weight')
    parser.add_argument("--reco_w", type=float, default=1.0, help = 'Reconstruction loss weight')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='-1', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')
    #visdom params
    parser.add_argument('-d','--disp_iters', type=int, default=1, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom', type=str, nargs='?', default=None, const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument('--visdom_iters', type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument('--print_factors', type=int, default=0, help='Set to 1 to visualize the anatomy factors in Visdom')

    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    #create and init device
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))    
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Training {0} for {1} epochs using a batch size of {2} on {3}'.format(args.name, args.epochs, args.batch_size, device))

    torch.manual_seed(667)
    if device.type == 'cuda':
        torch.cuda.manual_seed(667)

    #Visdom setup
    visualizer = utils.visualization.NullVisualizer() if args.visdom is None\
        else utils.visualization.VisdomVisualizer(args.name, args.visdom, count=1)

    #Model selection and initialization
    model_params = {
        'width': 224,
        'height': 224,
        'ndf': 64,
        'norm': "batchnorm",
        'upsample': "nearest",
        'num_classes': 3,
        'anatomy_out_channels': args.anatomy_factors,
        'z_length': args.modality_factors,
        'num_mask_channels': 8,

    }
    model = models.get_model(args.model_name, model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)

    #load training set
    if device.type == 'cuda':
        loader = loaders.ACDCLoader(args.data_path)
        data = loader.load_labelled_data(0, 'training')
        print(data.images.shape, data.masks.shape)
        print(data.volumes(), len(data.volumes()))
        images = torch.from_numpy(data.images.astype(float))
        masks = torch.from_numpy(data.masks.astype(float))
    else: #for debugging on cpu purposes
        images = torch.FloatTensor(10, 1, 224, 224).uniform_(-1, 1)
        masks = torch.FloatTensor(10, 3, 224, 224).uniform_(0, 1)
        masks[masks < 0.5] = 0
        masks[masks >= 0.5] = 1

    #load validation set
    if device.type == 'cuda':
        vdata = loader.load_labelled_data(0, 'validation')
        print(vdata.images.shape, vdata.masks.shape)
        print(vdata.volumes(), len(vdata.volumes()))
        vimages = torch.from_numpy(vdata.images.astype(float))
        vmasks = torch.from_numpy(vdata.masks.astype(float))
    else: #for on debugging on cpu purposes
        vimages = torch.FloatTensor(2, 1, 224, 224).uniform_(-1, 1)
        vmasks = torch.FloatTensor(2, 3, 224, 224).uniform_(0, 1)
        vmasks[vmasks < 0.5] = 0
        vmasks[vmasks >= 0.5] = 1

    #loss initialization
    l1_distance = nn.L1Loss().to(device)

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), betas=(0.5,0.999), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)

    #loss initialization
    total_loss = supervision.AverageMeter()
    running_reco_loss = supervision.AverageMeter()
    running_kl_loss = supervision.AverageMeter()
    running_dice_loss = supervision.AverageMeter()
    running_reg_loss = supervision.AverageMeter()
    running_focal_loss = supervision.AverageMeter()
    running_kl_a_loss = supervision.AverageMeter()
    running_reg_a_loss = supervision.AverageMeter()
    
    #validation loss initialization
    val_running_dice_loss = supervision.AverageMeter()

    #auxiliary tensors init
    b_images = torch.zeros(args.batch_size, 1, 224, 224)
    b_masks = torch.zeros(args.batch_size, masks.shape[1]+1, 224, 224)
    collapsed_b_masks = torch.zeros(args.batch_size, 1, 224, 224) #used in focal loss computation
    v_image = torch.zeros(1, 1, 224, 224)
    v_mask = torch.zeros(1, masks.shape[1]+1, 224, 224)
    total_batches = images.shape[0] // args.batch_size
    global_iterations = 0
    val_dice_list = []
    val_dice_best = 0.0
    
    #train/val process
    for epoch in range(args.epochs):
        idx = torch.randperm(images.shape[0])
        in_batch_iter = 0
        model.train()
        for iteration in range(images.shape[0]):
            if (iteration + args.batch_size) > images.shape[0]:
                break
            if in_batch_iter < args.batch_size:
                b_images[in_batch_iter] = images[idx[iteration]]
                #add an extra channel for mask background
                cmask = masks[idx[iteration]]
                logical_or = torch.sum(cmask, dim=0)
                tmask_0 = 1 - logical_or
                tmask = torch.cat([tmask_0.unsqueeze(0), cmask], dim=0)
                b_masks[in_batch_iter] = tmask
                in_batch_iter += 1
            else:
                #init batch-wise losses
                kl_loss = 0.0
                dice_loss = 0.0
                regression_loss = 0.0
                focal_loss = 0.0
                kl_a_loss = 0.0
                regression_a_loss = 0.0
                optimizer.zero_grad()
                #collapse mask tensor to 1 channel for the focal loss computation
                collapsed_b_masks = b_masks[:,1,:,:] + b_masks[:,2,:,:]*2 + b_masks[:,3,:,:]*3
                #forward pass
                reco, z_out, mu_tilde, a_mu_tilde, a_out, seg_pred, mu, logvar, a_mu, a_logvar = model(b_images.to(device), b_masks.to(device), 'training')
                #loss computation
                if args.charbonnier > 0:
                    l1_loss =  l1_distance(reco, b_images.to(device))
                    reco_loss = supervision.charbonnier_penalty(l1_loss)
                else:
                    reco_loss =  l1_distance(reco, b_images.to(device))
                if args.kl_w > 0.0:
                    kl_loss = supervision.KL_divergence(logvar, mu)
                if args.dice_w > 0.0:
                    dice_loss = supervision.dice_loss(seg_pred[:,1:,:,:], b_masks[:,1:,:,:].to(device))
                if args.regress_w > 0.0:
                    regression_loss = l1_distance(mu_tilde, z_out)
                if args.focal_w > 0.0:
                    collapsed_b_masks[collapsed_b_masks > 3] = 3
                    focal_loss = supervision.FocalLoss(gamma=2,alpha=0.25)(seg_pred, collapsed_b_masks.to(device))
                if args.model_name == 'sdnet2' and args.kl_w > 0.0:
                    kl_a_loss = supervision.KL_divergence(a_logvar, a_mu)
                if args.model_name == 'sdnet2' and args.regress_w > 0.0:
                    regression_a_loss = l1_distance(a_mu_tilde, a_out)
                if args.model_name == 'sdnet3' and args.kl_w > 0.0:
                    kl_a_loss = supervision.KL_divergence(a_logvar, a_mu)

                batch_loss = args.reco_w * reco_loss\
                                 + args.kl_w * kl_loss\
                                 + args.dice_w * dice_loss\
                                 + args.regress_w * regression_loss\
                                 + args.focal_w * focal_loss\
                                 + args.kl_w * kl_a_loss\
                                 + args.regress_w * regression_a_loss #2 last losses used only for the VAE anatomy encoding case

                #backprop and optimizer update
                batch_loss.backward()
                optimizer.step()
                #logging
                total_loss.update(batch_loss)
                running_reco_loss.update(reco_loss)
                running_kl_loss.update(kl_loss)
                running_dice_loss.update(dice_loss)
                running_reg_loss.update(regression_loss)
                running_focal_loss.update(focal_loss)
                running_kl_a_loss.update(kl_a_loss)
                running_reg_a_loss.update(regression_a_loss)

                #visualizations
                if (iteration + 1) % args.visdom_iters == 0 and args.visdom is not None:
                    visualizer.show_map(b_images.to(device), 'Input Image')
                    visualizer.show_map(reco, 'Reconstructed Image')
                    visualizer.show_seg_map(b_masks.to(device), 'GT Mask')
                    visualizer.show_seg_map(seg_pred, 'Predicted Mask')
                    if args.print_factors:
                        visualizer.show_anatomical_factors(a_out, 'Anatomical Factor')

                if (iteration + 1) % args.disp_iters <= args.batch_size:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    print("Epoch: {}, iteration: {}/{}\nLR: {}\nFocal: {}\nDice: {}\nReco: {}\nKLD: {}\nReg: {}\nKLD_a: {}\nReg_a: {}\nTotal average loss: {}\n\n"\
                        .format(epoch, iteration, images.shape[0], lr, running_focal_loss.avg, running_dice_loss.avg, running_reco_loss.avg,\
                            running_kl_loss.avg, running_reg_loss.avg, running_kl_a_loss.avg, running_reg_a_loss.avg, total_loss.avg))

                    #loss plots
                    if args.visdom is not None:
                        visualizer.append_loss(epoch, global_iterations, total_loss.avg.item(), "Total")
                        visualizer.append_loss(epoch, global_iterations, running_reco_loss.avg.item(), "Reconstruction")
                        visualizer.append_loss(epoch, global_iterations, running_focal_loss.avg.item(), "Focal")
                        visualizer.append_loss(epoch, global_iterations, running_kl_loss.avg.item(), "KLD")
                        visualizer.append_loss(epoch, global_iterations, running_dice_loss.avg.item(), "Dice")
                        visualizer.append_loss(epoch, global_iterations, running_reg_loss.avg.item(), "Regression")
                        visualizer.append_loss(epoch, global_iterations, running_kl_a_loss.avg.item(), "KLD_a")
                        visualizer.append_loss(epoch, global_iterations, running_reg_a_loss.avg.item(), "Regression_a")

                    total_loss.reset()
                    running_reco_loss.reset()
                    running_kl_loss.reset()
                    running_dice_loss.reset()
                    running_reg_loss.reset()
                    running_focal_loss.reset()
                    running_kl_a_loss.reset()
                    running_reg_a_loss.reset()
                global_iterations += args.batch_size
                in_batch_iter = 0
        #validation
        with torch.no_grad():
            model.eval()
            for iteration in range(vimages.shape[0]):
                v_image[0] = vimages[iteration]
                #add an extra channel for mask background
                cmask = vmasks[iteration]
                logical_or = torch.sum(cmask, dim=0)
                tmask_0 = 1 - logical_or
                tmask = torch.cat([tmask_0.unsqueeze(0), cmask], dim=0)
                v_mask[0] = tmask
                
                #forward pass
                _, _, _, _, _, seg_pred, _, _, _, _ = model(v_image.to(device), v_mask.to(device), 'val')                
                #dice score computation
                dice_loss = supervision.dice_score(seg_pred[:,1:,:,:], v_mask[:,1:,:,:].to(device))
                #logging
                val_running_dice_loss.update(dice_loss)

            print("Epoch: {},\nValidation Samples: {}\nDice: {}\n"\
                .format(epoch, iteration, val_running_dice_loss.avg))

            #val loss plots
            if args.visdom is not None:
                visualizer.append_loss(epoch, epoch, val_running_dice_loss.avg.item(), "Validation Dice")

            #check for plateau
            val_dice_curr = val_running_dice_loss.avg.item()
            scheduler.step(val_dice_curr)

            #save checkpoint for the best validation accuracy
            if val_dice_curr > val_dice_best: 
                val_dice_best = val_dice_curr
                print("Epoch checkpoint")
                current_dir = os.getcwd()
                final_dir = os.path.join(current_dir, args.save_path)
                utils.save_network_state(model, model_params['width'], model_params['height'], model_params['ndf'], \
                                    model_params['norm'], model_params['upsample'], model_params['num_classes'], \
                                    model_params['anatomy_out_channels'], model_params['z_length'],\
                                    model_params['num_mask_channels'], optimizer, \
                                    epoch, args.name + "_model_state_epoch_" + str(epoch), \
                                    final_dir)
            val_running_dice_loss.reset()

        

