import sys
from torchmetrics import MeanAbsoluteError
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
import wandb

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
        NiftiDataset.Resample(opt.new_resolution, opt.resample),
        NiftiDataset.Augmentation(),
        NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
        NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    ]

    train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True,
                             train=True)
    print('length train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=True)  # Here are then fed to the network with a defined batch size

    # WandB
    experiment = wandb.init(project='3D-CycleGan', resume='allow', anonymous='must', entity="3dcyclegan")
    # experiment = wandb.init(project='3D-CycleGan', resume=True, id= '') # if resuming (--continue_train)
    experiment.config.update(dict(epochs=opt.niter + opt.niter_decay, batch_size=opt.batch_size, learning_rate=opt.lr),
                             allow_val_change=True)

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            experiment.log({
                'train loss': model.get_current_losses(),
                'step': total_steps,
                'epoch': epoch})

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                t_data = iter_start_time - iter_data_time
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                mae = MeanAbsoluteError()
                val_score = mae(model.get_current_visuals()['rec_A'].cpu(), model.get_current_visuals()['real_A'].cpu())
                experiment.log({
                    'learning rate': model.optimizers[0].param_groups[0]['lr'],
                    'mae': val_score,
                    'epoch': epoch,
                    'step': total_steps,
                    'Image': wandb.Image(model.get_current_visuals()['real_A'].squeeze().data.cpu().numpy()[:, :, 32]),
                    'Labels': {
                        'true': wandb.Image(
                            model.get_current_visuals()['rec_A'].squeeze().data.cpu().numpy()[:, :, 32]),  # recreated
                        'pred': wandb.Image(
                            model.get_current_visuals()['fake_B'].squeeze().data.cpu().numpy()[:, :, 32]),  # t2 for now
                    },
                    'Masks': {
                        'true': wandb.Image(
                            model.get_current_visuals()['mask_A'].squeeze().data.cpu().numpy()[:, :, 32]),
                        'pred': wandb.Image(
                            model.get_current_visuals()['mask_B'].squeeze().data.cpu().numpy()[:, :, 32])}
                })

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
