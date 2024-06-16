import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataset import Vimeo90k
from evaluation.validation import recon_validate as validate
from synthesis import SynthesisNet
from flow import getFlowModel


def get_exp_cfg():
    parser = ArgumentParser()
    parser.add_argument('--name', default=None, required=True, help='name of the experiment to load.')
    parser.add_argument('--component', default='synthesis', choices=['synthesis', 'flow_teacher'], help='the component to evaluate.')
    parser.add_argument('--seed', type=int, default=0, help='random seed setting')
    parser.add_argument('--dataroot', type=str, default='/dataset', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_save_fig', default=10, help='number of batches to save as image during validation.')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='batch size to use for evaluation.')
    parser.add_argument('--logging', action='store_true', help='use logging on tensorboard')

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
    )
    return model


def evaluate():
    args = get_exp_cfg()
    
    # paths
    proj_dir = f'./experiments/{args.component}/{args.name}'
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
    valid_data = Vimeo90k(path=os.path.join(args.dataroot, 'vimeo_triplet'), is_train=False)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # load pretrained weights
    synth_model = build_synth(args)
    flow_model = getFlowModel(args.flow_arch)
    accelerator.print('loading checkpoints...')
    assert os.path.exists(f'{save_path}/model.pth'), 'path to model checkpoints do not exist!'
    ckpt = torch.load(f'{save_path}/model.pth', map_location='cpu')
    if 'synth_model' in ckpt:
        synth_model.load_state_dict(ckpt['synth_model'])
    if 'flow_model' in ckpt:
        flow_model.load_state_dict(ckpt['flow_model'])
    del ckpt

    # prepare accelerator
    if log_tracker is not None:
        accelerator.init_trackers('logs')
    synth_model, flow_model, valid_loader = accelerator.prepare(synth_model,  flow_model, valid_loader)
    log_tracker = accelerator.get_tracker('tensorboard') if args.logging else None

    # for evaluation
    set_seed(args.seed, device_specific=True)  # overwrite seeds
    _ = validate(
        synth_model,
        flow_model,
        valid_loader,
        999,
        accelerator,
        tracker=log_tracker,
        n_save_fig=args.n_save_fig,
    )
    accelerator.wait_for_everyone()
    if args.logging:
        accelerator.end_training()
    accelerator.print('evaluation finished.')
    return



if __name__ == '__main__':
    evaluate()