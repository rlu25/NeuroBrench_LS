import nibabel as nib
import numpy as np

def save_nifti(volume, affine, path):
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, path)

def mask_background(pred, reference):
    mask = reference > 0
    return pred * mask

def train():
    import time
    from options.train_options import TrainOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    import torch
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    model = create_model(opt)
    
    #Loading data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    dataset_size = len(data_loader)
    print('Training images = %d' % dataset_size)  
    
    # Load validation data
    opt.phase = 'val'
    val_data_loader = CreateDataLoader(opt)
    val_dataset = val_data_loader.load_data()
    print('Validation images = %d' % len(val_dataset))
    visualizer = Visualizer(opt)

    total_steps = 0
    #Starts training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            #Save current images (real_A, real_B, fake_A, fake_B, rec_A, rec_B)
            if  epoch_iter == opt.batchSize:
                save_result = total_steps % opt.update_html_freq == 0
                visuals, tensors = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch,epoch_iter, save_result)

            #Save current errors   
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
            #Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    
            iter_data_time = time.time()
            
        #Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        
        # Run validation
        print("Running validation...")
        val_errors = []
        for val_data in val_dataset:
            model.set_input(val_data)
            model.test()
            val_errors.append(model.get_current_errors())

        if val_errors:
            avg_errors = {k: sum(e[k] for e in val_errors) / len(val_errors) for k in val_errors[0]}
            visualizer.print_current_errors(epoch, epoch_iter, avg_errors, 0, 0)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, avg_errors)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()



def test():
    import sys
    sys.argv=args  
    
    import os
    from options.test_options import TestOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    from util import html
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse
    import numpy as np
    from collections import defaultdict
    import nibabel as nib
    import pandas as pd


    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # Load metadata
    metadata_csv = os.path.join(opt.dataroot, opt.phase, "slice_metadata.csv")
    metadata_df = pd.read_csv(metadata_csv)
    center_slice_dict = metadata_df.groupby('subject_id')['slice_index'].median().astype(int).to_dict()
    print(f"✅ Loaded metadata: {metadata_csv}")

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s' % (opt.phase))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' % (opt.name, opt.phase))
    metrics_path = os.path.join(web_dir, 'metrics_subjectwise.csv')

    # Prepare output CSV
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    pd.DataFrame(columns=['subject_id', 'psnr_B', 'ssim_B', 'mse_B', 'psnr_A', 'ssim_A', 'mse_A']).to_csv(metrics_path, index=False)

    img_path_list = []
    
    # Group slices by subject for 3D volume reconstruction
    generated_volumes_B = defaultdict(list)
    real_volumes_B = defaultdict(list)
    generated_volumes_A = defaultdict(list)
    real_volumes_A = defaultdict(list)
    
    current_subject = None
    import nibabel as nib
    n = 0
    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        model.set_input(data)
        model.test()
        visuals, tensors = model.get_current_visuals()

        img_path = model.get_image_paths()
        
        subject_id = metadata_df.loc[i, 'subject_id']
        if subject_id != current_subject:
            n = n + 1
            current_subject = subject_id
        
        slice_index = metadata_df.loc[i, 'slice_index']
        img_path[0] = f"{subject_id}_slice{slice_index:03d}"
        img_path_list.append(img_path)
        
        
        if slice_index == center_slice_dict[subject_id]:
            print('%04d: process image... %s' % (n, img_path))
            visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
            webpage.save()
        # 3D NIfTI saving and evaluation
        # Extract and normalize volumes (assuming shape is C, D, H, W for 3D data)
        # Extract volumes and normalize
        real_b = tensors['real_B'][0].cpu().numpy()[0]
        fake_b = tensors['fake_B'][0].cpu().numpy()[0]
        real_b = (real_b - real_b.min()) / (real_b.max() - real_b.min() + 1e-8)
        fake_b = (fake_b - fake_b.min()) / (fake_b.max() - fake_b.min() + 1e-8)
        real_volumes_B[subject_id].append(real_b)
        generated_volumes_B[subject_id].append(fake_b)



        # Optional: Also process A channel if needed
        if 'real_A' in tensors and 'fake_A' in tensors:
            real_a = tensors['real_A'][0].cpu().numpy()[0]
            fake_a = tensors['fake_A'][0].cpu().numpy()[0]
            real_a = (real_a - real_a.min()) / (real_a.max() - real_a.min() + 1e-8)
            fake_a = (fake_a - fake_a.min()) / (fake_a.max() - fake_a.min() + 1e-8)
            real_volumes_A[subject_id].append(real_a)
            generated_volumes_A[subject_id].append(fake_a)
        
        # After last slice of a subject
        is_last_slice = slice_index == metadata_df[metadata_df['subject_id'] == subject_id]['slice_index'].max()
        nii_dir = os.path.join(web_dir, 'nii')
        
        if not os.path.exists(nii_dir):
            os.makedirs(nii_dir)
        if is_last_slice:
            print(f"Completed subject: {subject_id}")
            pred_b = np.stack(generated_volumes_B[subject_id], axis=-1)
            real_b = np.stack(real_volumes_B[subject_id], axis=-1)
            
            
            affine = np.eye(4)  # or replace with actual affine if available
            pred_b = mask_background(pred_b, real_b)
            save_nifti(pred_b, affine, os.path.join(nii_dir, f"{subject_id}_fake_B.nii.gz"))
            save_nifti(real_b, affine, os.path.join(nii_dir, f"{subject_id}_real_B.nii.gz"))
            psnr_b = psnr(real_b, pred_b, data_range=1.0)
            ssim_b = ssim(real_b, pred_b, data_range=1.0)
            mse_b = mse(real_b, pred_b)

            psnr_a = ssim_a = mse_a = np.nan
            if subject_id in generated_volumes_A:
                pred_a = np.stack(generated_volumes_A[subject_id], axis=-1)
                real_a = np.stack(real_volumes_A[subject_id], axis=-1)
                pred_a = mask_background(pred_a, real_a)
                save_nifti(pred_a, affine, os.path.join(nii_dir, f"{subject_id}_fake_A.nii.gz"))
                save_nifti(real_a, affine, os.path.join(nii_dir, f"{subject_id}_real_A.nii.gz"))
                psnr_a = psnr(real_a, pred_a, data_range=1.0)
                ssim_a = ssim(real_a, pred_a, data_range=1.0)
                mse_a = mse(real_a, pred_a)

            pd.DataFrame([{
                'subject_id': subject_id,
                'psnr_B': psnr_b,
                'ssim_B': ssim_b,
                'mse_B': mse_b,
                'psnr_A': psnr_a,
                'ssim_A': ssim_a,
                'mse_A': mse_a,
            }]).to_csv(metrics_path, mode='a', header=False, index=False)

    webpage.save()
    print(f"Saved subject-wise metrics to: {metrics_path}")   


import sys
sys.argv.extend(['--model','cGAN','--no_dropout'])

args=sys.argv


if '--training' in str(args):
    train()
else:
    sys.argv.extend(['--serial_batches'])
    test()    