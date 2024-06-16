import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from math import exp
from torch.nn.functional import conv3d, pad

class PSNR:
    def __call__(self, pred, gt):
        b = gt.shape[0]
        se = (pred - gt) ** 2
        mse = torch.mean(se.reshape(b, -1), dim=1)
        return 10 * torch.log10((255. ** 2) / mse)


# based on UPRNet repo.
# ssim_matlab.
class SSIM:
    def __init__(self, window_size=11, window=None, size_average=False, full=False, val_range=None) -> None:
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window_3d(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
        window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous()
        return window
    
    def __call__(self, img1, img2):
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range

        padd = 0
        (_, _, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window_3d(real_size, channel=1).to(img1.device)
            # Channel is set to 1 since we consider color images as volumetric images
        else:
            window = self.window

        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)

        mu1 = conv3d(pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
        mu2 = conv3d(pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv3d(pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
        sigma2_sq = conv3d(pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
        sigma12 = conv3d(pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1).mean(1)

        if self.full:
            return ret, cs
        return ret


@ torch.no_grad()
class Evaluate:
    def __init__(self) -> None:
        """
        input must be in the range of [0, 1]
        """
        self.get_psnr = PSNR()
        self.get_ssim = SSIM()
        self.get_lpips = LPIPS().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_dists = DISTS().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = 255.0

    def to_uint8(self, img):
        img = img * self.scaler
        img = img.round()  # quantize
        img = torch.clamp(img, 0, 255)
        return img
    
    def uint8_to_float32(self, img):
        img = img / self.scaler
        return img

    @ torch.no_grad()
    def __call__(self, pred, gt):
        pred_uint8, gt_uint8 = self.to_uint8(pred), self.to_uint8(gt)
        pred = self.uint8_to_float32(pred_uint8)  # quantized pred values. (GT is already quantized. No need to do this.)
        scores = dict()
        scores['psnr'] = self.get_psnr(pred_uint8, gt_uint8)
        scores['ssim'] = self.get_ssim(pred, gt)
        scores['lpips'] = self.get_lpips(pred, gt, normalize=True).squeeze(-1).squeeze(-1).squeeze(-1)
        scores['dists'] = self.get_dists(pred, gt)
        return scores

