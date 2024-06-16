import os
import torch
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torchvision.utils import flow_to_image
from utils import set_mode
from evaluation.metrics import Evaluate


@ torch.no_grad()
def vfi_validate(
    model,
    dataloader=None,
    ep=0,
    accelerator=None,
    tracker=None,
    visualize_flows=False,
    n_save_fig=10,
    save_as_png=False,
    save_dir=None,
    **kwargs,
):
    if save_as_png:
        assert save_dir is not None, 'save_path must be configured to save all the final results.'
    set_mode(model, mode='eval')
    scores = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'dists': 0}
    n_samples = 0

    get_scores = Evaluate()
    torch.cuda.empty_cache()
    for i, data in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process)):
        input_frames, target_frames, target_t, name = data

        # motion modeling + synthesis
        pred, flows = model(input_frames, **kwargs)

        # compute scores
        current_scores = get_scores(pred, target_frames)
        current_scores = accelerator.gather_for_metrics(current_scores)
        n_samples += current_scores['psnr'].shape[0]
        for key, value in current_scores.items():
            scores[key] += value.sum()

        # save image results
        if n_save_fig == 'all' or i < n_save_fig:
            pred = accelerator.gather_for_metrics(pred.contiguous())
            name = accelerator.gather_for_metrics(name)
            if visualize_flows:
                flows = accelerator.gather_for_metrics(flows)
                flowt0, flowt1 = flows.chunk(2, dim=1)
                flowt0 = flow_to_image(flowt0).float() / 255
                flowt1 = flow_to_image(flowt1).float() / 255
            
            # saving operation on main process.
            if accelerator.is_main_process:
                # if save as png
                if save_as_png:
                    for b in range(pred.shape[0]):
                        filename = name[b].split('/')[-1].split('.')[0]
                        cur_filedir = os.path.join(save_dir,  *name[b].split('/')[:-1])
                        os.makedirs(cur_filedir, exist_ok=True)
                        to_pil_image(pred[b].cpu()).save(f'{cur_filedir}/{filename}_RGB.png')
                        if visualize_flows:
                            to_pil_image(flowt0[b].cpu()).save(f'{cur_filedir}/{filename}_flowt0.png')
                            to_pil_image(flowt1[b].cpu()).save(f'{cur_filedir}/{filename}_flowt1.png')
                # if save on tracker (tensorboard)
                elif tracker is not None:
                    tracker.log_images({f'Validation Batch {i:04d} Pred': pred}, step=ep)
                    if visualize_flows:
                        tracker.log_images({f'Validation Batch {i:04d} Flows/Flowt0': flowt0}, step=ep)
                        tracker.log_images({f'Validation Batch {i:04d} Flows/Flowt1': flowt1}, step=ep)

    # averaging
    for key, value in scores.items():
        scores[key] = value / n_samples

    # print evaluation result
    out_log = f'Validation at {ep} === '
    for key in scores:
        if tracker is not None:
            tracker.log({f'Eval/{key.upper()}': scores[key]}, step=ep)
        out_log = out_log + f'{key.upper()}: {scores[key].item():.4f}\t'
    accelerator.print(out_log)
    torch.cuda.empty_cache()
    set_mode(model, mode='train')
    return scores


@ torch.no_grad()
def recon_validate(
    synth_model,
    flow_model,
    dataloader=None,
    ep=0,
    accelerator=None,
    tracker=None,
    n_save_fig=10,
    **kwargs,
):
    set_mode(synth_model, flow_model, mode='eval')
    scores = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'dists': 0}
    n_samples = 0

    get_scores = Evaluate()
    torch.cuda.empty_cache()
    for i, data in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process)):
        input_frames, target_frames, target_t, name = data

        # forwarding
        flows = flow_model(torch.cat([target_frames, target_frames], dim=0), rearrange(input_frames, 'b c f h w -> (f b) c h w'))
        flows = rearrange(flows, '(f b) c h w -> b (f c) h w', f=2)
        pred = synth_model(input_frames, flows)

        # compute scores
        current_scores = get_scores(pred, target_frames)
        current_scores = accelerator.gather_for_metrics(current_scores)
        n_samples += current_scores['psnr'].shape[0]
        for key, value in current_scores.items():
            scores[key] += value.sum()

        # save image results
        if n_save_fig == 'all' or i < n_save_fig:
            pred = accelerator.gather_for_metrics(pred.contiguous())
            if accelerator.is_main_process and tracker is not None:
                tracker.log_images({f'Validation Batch {i:04d} Pred': pred}, step=ep)
    
    # averaging
    for key, value in scores.items():
        scores[key] = value / n_samples

    # print evaluation result
    out_log = f'Validation at {ep} === '
    for key in scores:
        if tracker is not None:
            tracker.log({f'Eval/{key.upper()}': scores[key]}, step=ep)
        out_log = out_log + f'{key.upper()}: {scores[key].item():.4f}\t'
    accelerator.print(out_log)
    torch.cuda.empty_cache()
    set_mode(synth_model, flow_model, mode='train')
    return scores
