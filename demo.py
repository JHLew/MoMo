import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataset import VideoData
from synthesis import SynthesisNet
from diffusion.momo import MoMo
from tqdm import tqdm
from utils import set_mode, frames2video, tensor2opencv
import shutil


def get_exp_cfg():
    parser = ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='path to the video to conduct frame interpolation.')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the interpolated result.')
    parser.add_argument('--ckpt_path', type=str, default='./experiments/diffusion/momo_full/weights/model.pth', help='path to the pretrained model weights')
    parser.add_argument('--use_png_buffer', action='store_true', help='save the extracted frames as png as buffer for processing, in case memory is insufficient.')
    parser.add_argument('--seed', type=int, default=42, help='random seed setting')
    parser.add_argument('--mp', type=str, default='no', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=2)
    
    # inference parameters
    parser.add_argument('--inf_steps', type=int, default=8, help='number of denoising steps to use for inference.')
    parser.add_argument('--resize_to_fit', action='store_true', help='fit to training resolution and resize back to input resolution for inference.')
    parser.add_argument('--no_resize_inf', action='store_false', dest='resize_to_fit')
    parser.set_defaults(resize_to_fit=True)
    parser.add_argument('--pad_to_fit_unet', action='store_true', help='avoid errors in resolution mismatch after a sequence of downsamplings and upsamplings in the U-Net by padding vs resizing')
    parser.add_argument('--resize_to_fit_unet', action='store_false', dest='pad_to_fit_unet')
    parser.set_defaults(pad_to_fit_unet=False)
    args = parser.parse_args()

    return args


@ torch.no_grad()
def run():
    args = get_exp_cfg()
    accelerator = Accelerator(
        mixed_precision=args.mp,
        split_batches=False,
    )

    # print setting
    accelerator.print('\n\n#######################################################################################\n')
    accelerator.print(f'x2 interpolation on <{args.video}>\n')
    accelerator.print(args)
    accelerator.print('\n#######################################################################################\n\n')

    # load pretrained models
    synth_model = SynthesisNet()
    model = MoMo(synth_model=synth_model)
    assert os.path.exists(args.ckpt_path), 'path to model checkpoints do not exist!'
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    param_ckpt = ckpt['model']
    model.load_state_dict(param_ckpt)
    del ckpt

    # dataloader setting
    video_data = VideoData(args.video, args.use_png_buffer)
    dataloader = DataLoader(video_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # prepare accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # run interpolation
    set_seed(args.seed, device_specific=True)
    set_mode(model, mode='eval')
    output_frames_list = []
    torch.cuda.empty_cache()
    for data in tqdm(dataloader, disable=not accelerator.is_main_process):
        frame0, frame1 = data
        pred, _ = model(
            torch.stack([frame0, frame1], dim=2),
            num_inference_steps=args.inf_steps,
            resize_to_fit=args.resize_to_fit,
            pad_to_fit_unet=args.pad_to_fit_unet,
        )
        pred = accelerator.gather_for_metrics(pred.contiguous())
        frame0 = accelerator.gather_for_metrics(frame0)
        
        # save results and connect the frames
        if accelerator.is_main_process:
            for b in range(pred.shape[0]):
                output_frames_list.append(tensor2opencv(frame0[b].cpu()))
                output_frames_list.append(tensor2opencv(pred[b].cpu()))
    # don't forget to include the very last frame
    frame1 = accelerator.gather_for_metrics(frame1)
    output_frames_list.append(tensor2opencv(frame1[-1].cpu()))

    # convert to video format
    if accelerator.is_main_process:
        frames2video(args.output_path, output_frames_list)
        if args.use_png_buffer:
            shutil.rmtree('./frames_buffer')
    # finish process
    accelerator.wait_for_everyone()
    accelerator.print('interpolation finished.')
    return
    

if __name__ == '__main__':
    run()
