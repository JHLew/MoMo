import torch
import torch.nn as nn
from torch.nn.functional import pad, conv2d, l1_loss
import numpy as np


class LapLoss(nn.Module):
    # modified from
    # https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8
    def __init__(self, max_levels=5, channels=3):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_levels = max_levels
        self.kernel = self.build_gauss_kernel(channels=channels, device=self.device)

    def build_gauss_kernel(self, channels=3, device=torch.device('cpu')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = pad(img, (2, 2, 2, 2), mode='reflect')
        out = conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img, max_levels=3):
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def forward(self, input, target):
        pyr_input = self.laplacian_pyramid(img=input, max_levels=self.max_levels)
        pyr_target = self.laplacian_pyramid(img=target, max_levels=self.max_levels)
        return sum(l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class CharbonnierLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-3) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, x0, x1):
        diff = x0 - x1
        squared_sum = (diff ** 2) + self.eps
        loss = squared_sum ** self.alpha
        return loss.mean()


class CensusLoss(nn.Module):
    # modified from VFIformer.
    def __init__(self, patch_size=7, to_grayscale=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_channels = patch_size * patch_size
        self.patch_size = patch_size
        self.padding = int(patch_size / 2)
        self.to_grayscale = to_grayscale
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(self.device)  # mask with one 1 and others 0.

    def census_transform(self, x):
        patches = conv2d(x, self.w, padding=self.padding, bias=None)  # get 49 neighboring pixels of the each pixel.
        transf = patches - x  # compute differnece with center (anchor) pixel
        transf = transf / torch.sqrt(0.81 + transf ** 2)  # not sure why..
        return transf

    def rgb2gray(self, x):
        if self.to_grayscale:
            r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
            grayscale = grayscale.unsqueeze(1)
        else:
            grayscale = x.mean(dim=1, keepdim=True)
        return grayscale

    def hamming_distance(self, x0, x1):
        dist = (x0 - x1) ** 2
        dist_norm = dist / (0.1 + dist)  # not sure why..
        dist_norm = torch.mean(dist_norm, 1, keepdim=True)
        return dist_norm

    def valid_mask(self, x, padding):
        b, _, h, w = x.shape
        valid_regions = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).float().to(self.device)
        valid_mask = pad(valid_regions, [padding, padding, padding, padding])
        return valid_mask

    def forward(self, x0, x1):
        _x0 = self.census_transform(self.rgb2gray(x0))
        _x1 = self.census_transform(self.rgb2gray(x1))
        valid_mask = self.valid_mask(_x0, 1)
        loss = self.hamming_distance(_x0, _x1) * valid_mask
        return loss.mean()


class L1Census(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        if eps > 0:
            self.l1 = nn.L1Loss()
        else:
            self.l1 = CharbonnierLoss(eps=eps)
        self.census = CensusLoss()

    def forward(self, x0, x1):
        l1_loss = self.l1(x0, x1)
        census_loss = self.census(x0, x1)
        return l1_loss + census_loss


from lpips import LPIPS
from torchvision.models.vgg import vgg19, VGG19_Weights
class ReconLPIPSLoss(nn.Module):
    def __init__(self,
                 recon_loss='L1',
                 w_lpips=1.,
                 w_style=20.,
                 eps=0.,
                 **kwargs) -> None:
        super().__init__()
        if recon_loss == 'L1':
            self.recon_loss_fn = nn.L1Loss()
        elif recon_loss == 'MSE':
            self.recon_loss_fn = nn.MSELoss()
        elif recon_loss == 'Laplacian':
            self.recon_loss_fn = LapLoss()
        elif recon_loss == 'L1Census':
            self.recon_loss_fn = L1Census(eps=eps)
        else:
            raise NotImplementedError('no such reconstruction loss.')

        self.w_lpips = w_lpips
        self.w_style = w_style

        if self.w_lpips > 0:
            self.lpips = LPIPS().eval()
        if self.w_style > 0:
            # FILM paper's parameters.
            self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float().reshape(1, 3, 1, 1)
            self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).float().reshape(1, 3, 1, 1)
            self.alpha_l = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]  # parameters from FILM (Reda et al.)
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            self.vgg_feats = nn.ModuleList([vgg.features[:4], vgg.features[4:9], vgg.features[9:14], vgg.features[14:23], vgg.features[23:32]])

    def get_vgg_features(self, x):
        # assuming the input is in the range of [-1, 1]
        x = (x + 1) / 2
        x = (x - self.vgg_mean.to(x.device)) / self.vgg_std.to(x.device)
        feat1_2 = self.vgg_feats[0](x)
        feat2_2 = self.vgg_feats[1](feat1_2)
        feat3_2 = self.vgg_feats[2](feat2_2)
        feat4_2 = self.vgg_feats[3](feat3_2)
        feat5_2 = self.vgg_feats[4](feat4_2)
        feats = [feat1_2, feat2_2, feat3_2, feat4_2, feat5_2]
        return feats
    
    def get_gram(self, x):
        if not isinstance(x, list):
            x = [x]
        grams = []
        for feat_lvl in x:
            grams.append(torch.einsum('b c h w, b d h w -> b c d', feat_lvl / 255., feat_lvl / 255.))
        return grams
    
    def forward(self, x, target):
        # input is [-1, 1]
        loss = self.recon_loss_fn(x, target)
        if self.w_lpips > 0:
            lpips = self.lpips(x, target)
            loss = loss + lpips * self.w_lpips
        if self.w_style > 0:
            x_feats = self.get_vgg_features(x)
            target_feats = self.get_vgg_features(target)
            x_grams = self.get_gram(x_feats)
            target_grams = self.get_gram(target_feats)
            style_loss = 0
            for i in range(len(x_grams)):
                x_gram_lvl = x_grams[i]
                target_gram_lvl = target_grams[i]
                style_loss = style_loss + ((x_gram_lvl - target_gram_lvl) ** 2).mean() * self.alpha_l[i]
            loss = loss + style_loss * self.w_style
        return loss
