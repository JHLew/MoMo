import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataset import Vimeo90k, SNU_FILM, Xiph, Middlebury_others
from evaluation.validation import vfi_validate as validate
from synthesis import SynthesisNet
from diffusion.momo import MoMo


def get_exp_cfg():
    parser = ArgumentParser()
    # shared
    parser.add_argument('--name', default=None, required=True, help='name of the experiment to load.')
    parser.add_argument('--seed', type=int, default=100, help='random seed setting')
    parser.add_argument('--dataroot', type=str, default='/dataset', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--mp', type=str, default='no', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--logging', action='store_true', help='use logging on tensorboard')

    # validation options
    parser.add_argument('--n_save_fig', default=0, help='number of batches to save as image during validation.')
    parser.add_argument('--visualize_flows', action='store_true', help='whether to visualize and save the generated optical flow map (motion map).')
    parser.add_argument('--save_as_png', action='store_true', help='whether to save as png files. If set to False, will be saved in tensorboard logs.')
    parser.add_argument('--png_save_dir', type=str, default=None, help='path to the directory to save results as png.')
    
    # inference parameters
    parser.add_argument('--inf_steps', type=int, default=8, help='number of denoising steps to use for inference.')
    parser.add_argument('--resize_to_fit', action='store_true', help='fit to training resolution and resize back to input resolution for inference.')
    parser.add_argument('--no_resize_inf', action='store_false', dest='resize_to_fit')
    parser.set_defaults(resize_to_fit=True)
    parser.add_argument('--pad_to_fit_unet', action='store_true', help='avoid errors in resolution mismatch after a sequence of downsamplings and upsamplings in the U-Net by padding vs resizing')
    parser.add_argument('--resize_to_fit_unet', action='store_false', dest='pad_to_fit_unet')
    parser.set_defaults(pad_to_fit_unet=False)
    parser.add_argument('--valid_dataset', type=str, default='SNU_FILM_hard', help='dataset to use for evaluation.')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='batch size to use for evaluation.')

    # synthesis model
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
    parser.add_argument('--max_rel_offset', type=float, default=1., help='maximum rate of flow value with respect to resolution')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'squaredcos_cap_v2'])
    parser.add_argument('--train_res', type=int, default=256, help='the crop size (resolution) used during training.')
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


def evaluate():
    args = get_exp_cfg()

    # paths
    proj_dir = f'./experiments/diffusion/{args.name}'
    save_path = f'{proj_dir}/weights'

    # logging
    log_tracker = 'tensorboard' if args.logging else None
    
    # initialize accelerator.
    accelerator = Accelerator(
        mixed_precision=args.mp,
        split_batches=True,
        log_with=log_tracker,
        project_dir=proj_dir,
    )

    # initial setting
    accelerator.print('\n\n#######################################################################################\n')
    accelerator.print(f'Evaluation on <{args.name}>\n')
    accelerator.print(args)
    accelerator.print('\n#######################################################################################\n\n')

    # dataset
    if 'SNU_FILM' in args.valid_dataset:
        test_data = SNU_FILM(os.path.join(args.dataroot, 'SNU_FILM'), mode=args.valid_dataset.split('_')[-1])
    elif 'Middlebury' == args.valid_dataset:
        test_data = Middlebury_others(os.path.join(args.dataroot, 'Middlebury'))
    elif 'Xiph' in args.valid_dataset:
        assert args.valid_dataset in ['Xiph_2K', 'Xiph_4K']
        test_data = Xiph(os.path.join(args.dataroot, 'Xiph'), setting=args.valid_dataset.split('_')[-1])
    elif args.valid_dataset == 'vimeo_triplet':
        test_data = Vimeo90k(path=os.path.join(args.dataroot, 'vimeo_triplet'), is_train=False)
    valid_loader = DataLoader(test_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load pretrained models
    synth_model = build_synth(args)
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
        train_res=args.train_res,
    )
    assert os.path.exists(f'{save_path}/model.pth'), 'path to model checkpoints do not exist!'
    ckpt = torch.load(f'{save_path}/model.pth', map_location='cpu')
    param_ckpt = ckpt['model']
    model.load_state_dict(param_ckpt)
    del ckpt
    
    # prepare accelerator
    if log_tracker is not None:
        accelerator.init_trackers('logs')
    model, valid_loader = accelerator.prepare(model, valid_loader)
    log_tracker = accelerator.get_tracker('tensorboard') if args.logging else None
    
    # evaluation
    set_seed(args.seed, device_specific=True)  # overwrite seeds
    accelerator.print(f'Evaluation on {args.valid_dataset}...')
    _ = validate(
        model,
        valid_loader,
        999,
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
    accelerator.wait_for_everyone()
    if args.logging:
        accelerator.end_training()
    accelerator.print('evaluation finished.')
    return
    

if __name__ == '__main__':
    evaluate()
