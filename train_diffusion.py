import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch_ema import ExponentialMovingAverage
from dataset import Vimeo90k, SNU_FILM, Xiph, Middlebury_others
from evaluation.validation import vfi_validate as validate
from synthesis import SynthesisNet
from flow import getFlowModel
from diffusion.momo import MoMo
from utils import get_device, is_best_performance, save_cfg, set_mode


def get_exp_cfg():
    parser = ArgumentParser()
    # shared
    parser.add_argument('--name', default=None, required=True, help='name of the experiment to load.')
    parser.add_argument('--teacher', type=str, required=True, help='name of the flow teacher model.')
    parser.add_argument('--resume', type=int, default=0, help='the epoch number to continue training from.')
    parser.add_argument('--seed', type=int, default=80, help='random seed setting')
    parser.add_argument('--dataroot', type=str, default='/dataset', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of total epochs to train.')
    parser.add_argument('--mp', type=str, default='no', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=8)

    # optimizing
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay in optimization.')
    parser.add_argument('--ema_decay_rate', type=float, default=0.9999, help='decay rate for exponential moving average of the model parameters.')
    parser.add_argument('--clip_grad', default='norm', choices=['norm', 'value', 'no'], help='gradient clipping method')
    parser.add_argument('--grad_max', type=float, default=1.0, help='maxiumum value for gradient clipping')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used in training.')
    parser.add_argument('--accum', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--crop_size', type=int, default=256, help='the crop size for training.')
    
    # validation options
    parser.add_argument('--n_save_fig', default=0, help='number of batches to save as image during validation.')
    parser.add_argument('--visualize_flows', action='store_true', help='whether to visualize and save the generated optical flow map (motion map).')
    parser.add_argument('--save_as_png', action='store_true', help='whether to save as png files. If set to False, will be saved in tensorboard logs.')
    parser.add_argument('--png_save_dir', type=str, default=None, help='path to the directory to save results as png.')
    
    # inference parameters
    parser.add_argument('--inf_steps', type=int, default=8, help='number of denoising steps to use for inference.')
    parser.add_argument('--resize_to_fit', action='store_true', help='whether to fit to the training resolution and resize back to input resolution for inference.')
    parser.add_argument('--no_resize_inf', action='store_false', dest='resize_to_fit')
    parser.set_defaults(resize_to_fit=True)
    parser.add_argument('--pad_to_fit_unet', action='store_true', help='avoid errors in resolution mismatch after a sequence of downsamplings and upsamplings in the U-Net by padding vs resizing')
    parser.add_argument('--resize_to_fit_unet', action='store_false', dest='pad_to_fit_unet')
    parser.set_defaults(pad_to_fit_unet=False)
    parser.add_argument('--valid_dataset', type=str, default='SNU_FILM_hard', help='dataset to use for validation.')
    parser.add_argument('--valid_batch_size', type=int, default=4, help='batch size to use for validation.')
    parser.add_argument('--valid_every', type=int, default=2, help='number of epochs per validation.')

    # experiment setting
    parser.add_argument('--metric', default='lpips', choices=['psnr', 'ssim', 'lpips', 'dists'], help='most important metric to use in saving ckpts.')

    # teacher, synth_model (synthesis) model
    parser.add_argument('--flow_arch', type=str, default='RAFT_Large', help='optical flow model architecture to use.')
    parser.add_argument('--s_dim', type=int, default=32)
    parser.add_argument('--s_recurrent_min_res', type=int, default=64)
    parser.add_argument('--s_norm_in', action='store_true')
    parser.add_argument('--s_no_norm_in', action='store_false', dest='s_norm_in')
    parser.set_defaults(s_norm_in=True)

    # diffusion model
    parser.add_argument('--dims', type=int, nargs='+', default=(256, 256, 512))
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--m_norm_in', action='store_true')
    parser.add_argument('--m_no_norm_in', action='store_false', dest='m_norm_in')
    parser.set_defaults(m_norm_in=True)
    parser.add_argument('--use_attn', action='store_true')
    parser.add_argument('--flow_scaler', type=int, default=128, help='denominator for normalization flow values.')
    parser.add_argument('--prediction_type', type=str, default='sample', choices=['epsilon', 'v_prediction', 'sample'])
    parser.add_argument('--align_corners', action='store_true')
    parser.add_argument('--padding', type=str, default='replicate', choices=['zeros', 'replicate', 'reflect', 'circular'])
    parser.add_argument('--interpolation', type=str, default='bicubic', choices=['nearest', 'bilinear', 'bicubic'])
    parser.add_argument('--clip_sample', action='store_true')
    parser.add_argument('--no_clip_sample', action='store_false', dest='clip_sample')
    parser.set_defaults(clip_sample=True)
    parser.add_argument('--max_rel_offset', type=float, default=1., help='maximum rate of flow value with respect to the resolution')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'squaredcos_cap_v2'])
    args = parser.parse_args()

    if args.n_save_fig != 'all':
        try:
            args.n_save_fig = int(args.n_save_fig)
        except:
            raise ValueError(f'n_save_fig argument must be \'all\' or an integer. Got {args.n_save_fig}')

    return args


def build_synth(args):
    model = SynthesisNet(
        latent_dim=args.s_dim,
        recurrent_min_res=args.s_recurrent_min_res,
        normalize_inputs=args.s_norm_in,
        align_corners=args.align_corners,
        padding=args.padding,
        interpolation=args.interpolation,
    )
    return model


def train():
    args = get_exp_cfg()
    device = get_device()

    # paths
    proj_dir = f'./experiments/diffusion/{args.name}'
    save_path = f'{proj_dir}/weights'
    
    # initialize accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum,
        mixed_precision=args.mp,
        split_batches=True,
        log_with='tensorboard',
        project_dir=proj_dir,
    )

    # save experimental configuration
    if accelerator.is_main_process:
        save_cfg(proj_dir, args)
    
    # initial setting
    set_seed(args.seed, device_specific=True)
    accelerator.print('\n\n#######################################################################################\n')
    accelerator.print(f'Experiment <{args.name}> starting from {args.resume}\n')
    accelerator.print(args)
    accelerator.print('\n#######################################################################################\n\n')

    # dataset
    train_data = Vimeo90k(path=os.path.join(args.dataroot, 'vimeo_triplet'), is_train=True, crop_size=args.crop_size)
    if args.valid_dataset == 'vimeo_triplet':
        test_data = Vimeo90k(path=os.path.join(args.dataroot, 'vimeo_triplet'), is_train=False)
    elif 'SNU_FILM' in args.valid_dataset:
        test_data = SNU_FILM(os.path.join(args.dataroot, 'SNU_FILM'), mode=args.valid_dataset.split('_')[-1])
    elif 'Middlebury' == args.valid_dataset:
        test_data = Middlebury_others(os.path.join(args.dataroot, 'Middlebury'))
    elif 'Xiph' in args.valid_dataset:
        assert args.valid_dataset in ['Xiph_2K', 'Xiph_4K']
        test_data = Xiph(os.path.join(args.dataroot, 'Xiph'), setting=args.valid_dataset.split('_')[-1])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(test_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # load models: synthesis model & teacher flow model
    synth_model = build_synth(args)
    teacher = getFlowModel(args.flow_arch)
    teacher_path = os.path.join('./experiments/flow_teacher', args.teacher, 'weights/model.pth')
    assert os.path.exists(teacher_path), 'path to pretrained optical flow model & synthsis model does not exist.'
    accelerator.print('loading flow teacher & synthesis model checkpoints...')
    teacher_ckpt = torch.load(teacher_path, map_location='cpu')
    synth_model.load_state_dict(teacher_ckpt['synth_model'])
    teacher.load_state_dict(teacher_ckpt['flow_model'])
    for params in synth_model.parameters():
        params.requires_grad = False
    for params in teacher.parameters():
        params.requires_grad = False
    del teacher_ckpt
    teacher.eval().to(device)

    # Motion Diffusion Model to train
    model = MoMo(
        synth_model=synth_model,
        dims=args.dims,
        T=args.T,
        flow_scaler=args.flow_scaler,
        prediction_type=args.prediction_type,
        align_corners=args.align_corners,
        clip_sample=args.clip_sample,
        max_rel_offset=args.max_rel_offset,
        beta_schedule=args.beta_schedule,
        use_attn=args.use_attn,
        norm_in=args.m_norm_in,
        padding=args.padding,
        interpolation=args.interpolation,
        train_res=args.crop_size,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # prepare accelerator
    accelerator.init_trackers('logs')
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    
    # Exponential Moving Averages
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay_rate)
    accelerator.register_for_checkpointing(ema)
    
    # best performance tracker
    best = 0 if args.metric in ['psnr', 'ssim'] else 100.0
    
    # if resume training
    if not args.resume == 0:
        accelerator.print(f'loading checkpoints...')
        accelerator.load_state(save_path)
        if os.path.exists(f'{save_path}/model.pth'):
            ckpt = torch.load(f'{save_path}/model.pth', map_location='cpu')
            best = ckpt['best']
            accelerator.print(f'previous best was: {best}')
            del ckpt
    
    # loss function.
    loss_fn = nn.L1Loss()

    # model save path
    os.makedirs(save_path, exist_ok=True)
    log_tracker = accelerator.get_tracker('tensorboard')

    # start training.
    ipe = int(np.ceil(int(len(train_loader)) / args.accum))
    accelerator.print('updates per epoch:', ipe)
    accelerator.print('start training.')
    cur_iters = args.resume * ipe
    ema.store()
    set_mode(model, mode='train')
    for epoch in range(args.resume, args.n_epochs):
        epoch_train_loss = 0
        model.train()
        optimizer.zero_grad()
        accelerator.print('\n\n==============================================================================\n')
        for _, data in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                input_frames, target_frames, _, _ = data
                
                # compute GT flow maps with the teacher flow model.
                with torch.no_grad():
                    teacher_GT = teacher(torch.cat([target_frames, target_frames], dim=0), rearrange(input_frames, 'b c f h w -> (f b) c h w'))
                    teacher_GT = rearrange(teacher_GT, '(f b) c h w -> b (f c) h w', f=2)
                
                # compute loss
                _loss = model(
                    input_frames,
                    target=teacher_GT,
                    target_frame=target_frames,
                    loss_fn=loss_fn
                ).mean()

                # update params
                accelerator.backward(_loss)
                if args.clip_grad != 'no':
                    if accelerator.sync_gradients:
                        if args.clip_grad == 'norm':
                            accelerator.clip_grad_norm_(model.parameters(), args.grad_max)
                        elif args.clip_grad == 'value':
                            accelerator.clip_grad_value_(model.parameters(), args.grad_max)
                optimizer.step()
                ema.update()
                optimizer.zero_grad()

                # logging
                with torch.no_grad():
                    avg_loss = accelerator.gather_for_metrics(_loss).mean()
                epoch_train_loss += avg_loss.item()
                accelerator.log({'Batch loss': avg_loss}, step=cur_iters)
                cur_iters += 1

        # after one epoch: log epoch loss
        epoch_train_loss /= ipe
        accelerator.print(f'At {epoch}. Train: {epoch_train_loss:.8f}')
        accelerator.print()  # spacing
        accelerator.log({'Epoch loss': epoch_train_loss}, step=epoch)
        accelerator.save_state(save_path)

        # validation
        if (epoch + 1) % args.valid_every == 0:
            ema.store()
            ema.copy_to()
            valid_scores = validate(
                model,
                valid_loader,
                epoch,
                accelerator,
                tracker=log_tracker,
                visualize_flows=args.visualize_flows,
                n_save_fig=args.n_save_fig,
                save_as_png=args.save_as_png,
                save_dir=args.png_save_dir,
                num_inference_steps=args.inf_steps,
                resize_to_fit=args.resize_to_fit,
                pad_to_fit_unet=args.pad_to_fit_unet,
            )
            # save if best performance
            is_best, best = is_best_performance(scores=valid_scores, prev_best=best, metric=args.metric)
            if is_best:
                accelerator.print('saving best weights...')
                ckpt = {
                    'model': accelerator.unwrap_model(model, keep_fp32_wrapper=True).state_dict(),
                    'best': best
                }
                accelerator.save(ckpt, f'{save_path}/model.pth')
            ema.restore()

    # end of training
    accelerator.wait_for_everyone()
    accelerator.print(f'end of training. Best performance is: {best}')
    accelerator.end_training()


if __name__ == '__main__':
    train()
