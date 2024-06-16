import torch
import os
import json
import cv2
import numpy as np


def get_device():
    device = 'cpu'
    if torch.cuda.is_available:
        device = 'cuda'
    elif torch.backends.mps.is_available:
        device = 'mps'
    return device


def is_best_performance(scores, prev_best, metric='lpips'):
    if metric in ['lpips', 'dists']:
        return scores[metric] < prev_best, min(scores[metric], prev_best)
    elif metric in ['psnr', 'ssim']:
        return scores[metric] > prev_best, max(scores[metric], prev_best)
    else:
        raise NotImplementedError('no such metric used for evaluation.')
    

def save_cfg(path, args):
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(f'{path}/args.txt'):
        with open(f'{path}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        assert args.resume > 0, f'Experiment of the same name already exists. Are you trying to resume training?'


def set_mode(*models, mode='eval'):
    assert mode in ['train', 'eval'], 'no such mode for models. Should be either train or eval mode.'
    if mode == 'train':
        for model in models:
            model.train()
    elif mode == 'eval':
        for model in models:
            model.eval()


def video2frames(video_path, output_path=None):
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = []
    if output_path is not None:
        video_name = os.path.basename(video_path).split('.')[0]
        os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
    for i in range(n_frames):
        ok, frame = video.read()
        if ok:
            if output_path is not None:
                frame_save_path = f'{output_path}/{video_name}/{i:06}.png'
                cv2.imwrite(frame_save_path, frame)
                frames_list.append(frame_save_path)
            else:
                frames_list.append(frame)  # frames are saved as cv2 format (BGR order).
        else:
            print(f'Error in frame no.:{i}')
    video.release()
    return frames_list


def frames2video(videopath, frames):
    # frames must be given in cv2 format (BGR order).
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(w, h))
    for frame in frames:
        assert h == frame.shape[0] and w == frame.shape[1], 'resolution inconsistency detected among frames. please check again.'
        writer.write(frame)
    writer.release()


def tensor2opencv(x):
    x = np.round(np.clip(x.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


if __name__ == '__main__':
    clip = './figures/SquareAndTimelapse_input.mp4'
    frames_list = video2frames(clip)
    import imageio
    filename = f'{clip[:-4]}.gif'
    for i, frame in enumerate(frames_list):
        frames_list[i] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, 270), interpolation=cv2.INTER_AREA)
    imageio.mimsave(filename, frames_list, fps=30)