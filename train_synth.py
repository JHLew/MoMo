import os
import torch
import torch.optim as optim
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch_ema import ExponentialMovingAverage
from dataset import Vimeo90k
from evaluation.validation import recon_validate as validate
from synthesis import SynthesisNet
from flow import getFlowModel
from loss import ReconLPIPSLoss
from utils import get_device, is_best_performance, save_cfg, set_mode


def get_exp_cfg():
    parser = ArgumentParser()
    # shared
    parser.add_argument('--name', default=None, required=True, help='name of the experiment to load.')
    parser.add_argument('--resume', type=int, default=0, help='the epoch number to continue training from.')
    parser.add_argument('--seed', type=int, default=42, help='random seed setting')
    parser.add_argument('--dataroot', type=str, default='/dataset', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--n_epochs', type=int, default=150, help='number of total epochs to train.')
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=8)
    
    # optimizing
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in optimization.')
    parser.add_argument('--ema_decay_rate', type=float, default=0.999, help='decay rate for exponential moving average of the model parameters.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size used in training.')
    parser.add_argument('--accum', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--crop_size', type=int, default=256, help='the crop size for training.')
    
    # validation
    parser.add_argument('--n_save_fig', default=10, help='number of batches to save as image during validation.')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='batch size to use for validation.')
    parser.add_argument('--valid_every', type=int, default=1, help='number of epochs per validation.')

    # experiment setting
    parser.add_argument('--metric', default='lpips', choices=['psnr', 'ssim', 'lpips', 'dists'], help='most important metric to use in saving ckpts.')
    parser.add_argument('--w_lpips', type=float, default=1)
    parser.add_argument('--w_style', type=float, default=20.)
    parser.add_argument('--loss_type', type=str, default='L1', choices=['L1', 'MSE', 'Laplacian', 'L1Census'], help='the base reconstruction loss to use.')
    parser.add_argument('--charb_eps', type=float, default=1e-6)
    parser.add_argument('--value_range', type=float, default=2.)

    # model
    parser.add_argument('--flow_arch', type=str, default='RAFT_Large', help='optical flow model architecture to use.')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--recurrent_min_res', type=int, default=64)
    parser.add_argument('--normalize_inputs', action='store_true')
    parser.add_argument('--no_normalize_inputs', action='store_false', dest='normalize_inputs')
    parser.set_defaults(normalize_inputs=True)
    parser.add_argument('--align_corners', action='store_true')
    parser.add_argument('--padding', type=str, default='replicate', choices=['zeros', 'replicate', 'reflect', 'circular'])
    parser.add_argument('--interpolation', type=str, default='bicubic', choices=['nearest', 'bilinear', 'bicubic'])
    parser.add_argument('--multi_scale_loss', action='store_true', help='whether to use supervision on multi scale reconstruction.')
    args = parser.parse_args()

    if args.n_save_fig != 'all':
        try:
            args.n_save_fig = int(args.n_save_fig)
        except:
            raise ValueError(f'n_save_fig argument must be \'all\' or an integer. Got {args.n_save_fig}')

    return args


def build_synth(args):
    model = SynthesisNet(
        latent_dim=args.latent_dim,
        recurrent_min_res=args.recurrent_min_res,
        normalize_inputs=args.normalize_inputs,
        align_corners=args.align_corners,
        padding=args.padding,
        interpolation=args.interpolation,
        multi_scale_loss=args.multi_scale_loss,
    )
    return model


def train():
    args = get_exp_cfg()
    device = get_device()
    
    # paths
    proj_dir = f'./experiments/synthesis/{args.name}'
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
    valid_data = Vimeo90k(path=os.path.join(args.dataroot, 'vimeo_triplet'), is_train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)    
    
    # models
    flow_model = getFlowModel(args.flow_arch).to(device).eval()
    for params in flow_model.parameters():
        params.requires_grad = False
    
    synth_model = build_synth(args)
    optimizer = optim.AdamW(synth_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # prepare accelerator
    accelerator.init_trackers('logs')
    synth_model, optimizer, train_loader, valid_loader = accelerator.prepare(synth_model, optimizer, train_loader, valid_loader)
    log_tracker = accelerator.get_tracker('tensorboard')

    # Exponential Moving Average
    ema = ExponentialMovingAverage(synth_model.parameters(), decay=args.ema_decay_rate, use_num_updates=True)
    accelerator.register_for_checkpointing(ema)

    # best performance tracker
    best = 0 if args.metric in ['psnr', 'ssim'] else 100.0

    # if resume training
    if not args.resume == 0:
        accelerator.print('loading checkpoints...')
        accelerator.load_state(save_path)
        if os.path.exists(f'{save_path}/model.pth'):
            ckpt = torch.load(f'{save_path}/model.pth', map_location='cpu')
            best = ckpt['best']
            accelerator.print(f'previous best was: {best}')
            del ckpt
    
    # loss function
    loss_fn = ReconLPIPSLoss(
        recon_loss=args.loss_type,
        w_lpips=args.w_lpips,
        w_style=args.w_style,
        _range=args.value_range,
        eps=args.charb_eps
    ).to(device)

    # model save path
    os.makedirs(save_path, exist_ok=True)

    # start training.
    ipe = int(np.ceil(np.ceil(len(train_loader) / args.accum)))
    accelerator.print('iterations per epoch:', ipe)
    accelerator.print('start training.')
    cur_iters = args.resume * ipe
    ema.store()
    set_mode(synth_model, mode='train')
    for epoch in range(args.resume, args.n_epochs):
        epoch_train_loss = 0
        synth_model.train()
        optimizer.zero_grad()
        accelerator.print('\n\n==============================================================================\n')
        for _, data in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            with accelerator.accumulate(synth_model):
                input_frames, target_frames, _, _ = data

                # get optical flows
                with torch.no_grad():
                    flows = flow_model(torch.cat([target_frames, target_frames], dim=0), rearrange(input_frames, 'b c f h w -> (f b) c h w'), final_only=True)
                    flows = rearrange(flows, '(f b) c h w -> b (f c) h w', f=2)
                
                # compute recon loss
                _loss = synth_model(
                    input_frames,
                    flows,
                    target=target_frames,
                    loss_fn=loss_fn,
                ).mean()

                # update params
                accelerator.backward(_loss)
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
        accelerator.save_state(save_path)
        accelerator.log({'Epoch loss': epoch_train_loss}, step=epoch)
        
        # validation
        if (epoch + 1) % args.valid_every == 0:
            ema.store()
            ema.copy_to()
            valid_scores = validate(
                synth_model,
                flow_model,
                valid_loader,
                epoch,
                accelerator,
                tracker=log_tracker,
                n_save_fig=args.n_save_fig,
            )

            # save if best performance
            is_best, best = is_best_performance(scores=valid_scores, prev_best=best, metric=args.metric)
            if is_best:
                accelerator.print('saving best weights...')
                ckpt = {
                    'synth_model': accelerator.unwrap_model(synth_model, keep_fp32_wrapper=True).state_dict(),
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