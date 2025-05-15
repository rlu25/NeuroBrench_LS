### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from models.models import create_model
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
from models.networks import define_D_3D as define_D
from util.image_pool import ImagePool
from models.networks import GANLoss
from models.pre_r3d_18 import Res3D
import math  # Add this import at the top
import glob


def lcm(a, b): return abs(a * b) // math.gcd(a, b) if a and b else 0


def discriminate(D, fake_pool, input_label, test_image, use_pool=False):
    input_concat = torch.cat((input_label, test_image.detach()), dim=1).to(opt.device)
    if use_pool:
        fake_query = fake_pool.query(input_concat)
        return D.forward(fake_query)
    else:
        return D.forward(input_concat)

def get_latest_checkpoint(path_pattern):
    checkpoint_list = glob.glob(path_pattern)
    if not checkpoint_list:
        return None
    # Sort by step number extracted from filename
    checkpoint_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('ckpt')[-1]))
    return checkpoint_list[-1]

train_csv = '../datasets/train_split.csv'
base_dir = "../../NeurIPS-Clean-T1T2"

opt = TrainOptions().parse()
opt.name = 'mae'
opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu")
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
opt.train_csv = train_csv
opt.base_dir = base_dir
opt.batchSize = 16
start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
# opt.debug = True
# if opt.debug:
#     opt.display_freq = 1
#     opt.print_freq = 1
#     opt.niter = 1
#     opt.niter_decay = 0
#     opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
ler = opt.lr
# mae = torch.nn.L1Loss()

mse = torch.nn.MSELoss()
# mse = torch.nn.L1Loss()
G = create_model(opt)
G.to(opt.device)

mae_dict = torch.load('/labs/wanglab/projects/NeurIPS-Clean-T1T2/other_for_all/MAE/pretrain_model/3D_MAE/model_final.pth')
G.enc.load_state_dict(mae_dict, strict=False)

G.train()
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.001)
visualizer = Visualizer(opt)

ResNet3D = Res3D()
ResNet3D.to(opt.device)


D = define_D(2, 64, 2, 'instance3D', False, 3, True, opt.gpu_ids)
D.to(opt.device)
# D.load_state_dict(torch.load('../ckpts_PTNet/PTNet_retrain_T12T2_0407/D_ckpt30101760.pth'))
D.train()
params = list(D.parameters())
optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.001)
fake_pool = ImagePool(0)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
CE = nn.CrossEntropyLoss()
criterionGAN = GANLoss(use_lsgan=not False, tensor=torch.cuda.FloatTensor)
criterionFeat = torch.nn.L1Loss()
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

print(G)

# Check if checkpoint exists and resume training
latest_G_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth')
latest_D_path = os.path.join(opt.checkpoints_dir, opt.name, 'D_latest.pth')

if not os.path.exists(latest_G_path):
    latest_G_path = get_latest_checkpoint(os.path.join(opt.checkpoints_dir, opt.name, 'ckpt*.pth'))
if not os.path.exists(latest_D_path):
    latest_D_path = get_latest_checkpoint(os.path.join(opt.checkpoints_dir, opt.name, 'D_ckpt*.pth'))


if os.path.exists(iter_path) and latest_G_path and latest_D_path:
    print(f"Loading checkpoints:\n  G: {latest_G_path}\n  D: {latest_D_path}")
    G.load_state_dict(torch.load(latest_G_path))
    D.load_state_dict(torch.load(latest_D_path))
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        print(f"Resuming from epoch {start_epoch}, iteration {epoch_iter}")
    except:
        print("Warning: Failed to load iter.txt, starting from scratch.")
        start_epoch, epoch_iter = 1, 0
else:
    print("No checkpoints found, training from scratch.")

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        # print(data['label'].shape)
        input_label = Variable(data['label'].to(opt.device))
        target_image = Variable(data['image'].to(opt.device))
        generated = G(input_label)
        loss_mse = mse(generated, target_image)

        # Fake Detection and Loss
        # for para in D.parameters():
        #     para.requires_grad = True
        pred_fake_pool = discriminate(D, fake_pool, input_label, generated, use_pool=True)
        loss_D_fake = criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = discriminate(D, fake_pool, input_label, target_image)
        loss_D_real = criterionGAN(pred_real, True)
        # for para in D.parameters():
        #     para.requires_grad = False
        # GAN loss (Fake Passability Loss)
        pred_fake = D.forward(torch.cat((input_label, generated), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        # GAN feature matching loss

        target_image = target_image.expand(-1, 3, -1, -1, -1)
        generated = generated.expand(-1, 3, -1, -1, -1)
        feat_res_real = ResNet3D(target_image)
        feat_res_fake = ResNet3D(generated)
        loss_G_GAN_ResNet = 0
        res_weights = [ 1.0/16, 1.0/8, 1.0/4, 1.0]
        feature_level = ['layer1', 'layer2', 'layer3', 'layer4']
        for tmp_i in range(len(feature_level)):
            loss_G_GAN_ResNet += criterionFeat(feat_res_real[feature_level[tmp_i]].detach(),feat_res_fake[feature_level[tmp_i]]) * res_weights[tmp_i]

        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (3 + 1)
        D_weights = 1.0 / 3
        for tpp_i in range(3):
            for tmp_j in range(len(pred_fake[tpp_i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    criterionFeat(pred_fake[tpp_i][tmp_j], pred_real[tpp_i][tmp_j].detach())

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = loss_mse * 100.0 + loss_G_GAN + loss_G_GAN_ResNet * 10.0 + loss_G_GAN_Feat * 10.0
        loss_dict = dict(zip(['MSE', 'G_GAN', 'G_GAN_ResNet', 'G_GAN_Feat','D_fake', 'D_real'], [loss_mse.item(), loss_G_GAN.item(),
                                                                                  loss_G_GAN_ResNet.item(),
                                                                                  loss_G_GAN_Feat.item(),
                                                                                  loss_D_fake.item(),
                                                                                  loss_D_real.item()]), )
        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0, :, :, :, 32], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0, :, :, :, 32])),
                                   ('real_image', util.tensor2im(data['image'][0, :, :, :, 32]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ## save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_latest.pth'))

        if epoch_iter >= dataset_size:
            break

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_ckpt%d%d.pth' % (epoch, total_steps)))

    ## linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = ler
            print('change lr to ')
            print(param_group['lr'])
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = ler