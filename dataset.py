from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch.utils.data as data
import os
from glob import glob
from random import randint
import torch
import numpy as np
import cv2
from utils import video2frames


class Vimeo90k(data.Dataset):
    def __init__(self, path, is_train=True, crop_size=None):
        super().__init__()
        train_list = os.path.join(path, 'tri_trainlist.txt')
        test_list = os.path.join(path, 'tri_testlist.txt')
        self.frame_list = os.path.join(path, 'sequences')
        self.is_train = is_train
        if self.is_train:
            triplet_list = train_list
        else:
            triplet_list = test_list
        with open(triplet_list) as triplet_list_file:
            triplet_list = triplet_list_file.readlines()
            triplet_list_file.close()
        self.triplet_list = triplet_list[:-1]
        if crop_size is None or crop_size <= 0:
            self.crop_size = None
        else:
            if type(crop_size) is not tuple:
                crop_size = (crop_size, crop_size)
            self.crop_size = crop_size

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        triplet_path = self.triplet_list[idx]
        if triplet_path[-1:] == '\n':
            triplet_path = triplet_path[:-1]
        try:
            vid_no, seq_no = triplet_path.split('/')
            name = vid_no + '_' + seq_no
        except:
            name = triplet_path
        triplet_path = os.path.join(self.frame_list, triplet_path)
        im1 = os.path.join(triplet_path, 'im1.png')
        im2 = os.path.join(triplet_path, 'im2.png')
        im3 = os.path.join(triplet_path, 'im3.png')

        # read image file
        im1 = Image.open(im1).convert('RGB')
        im2 = Image.open(im2).convert('RGB')
        im3 = Image.open(im3).convert('RGB')

        # data augmentation - random flip / sequence flip
        if self.is_train:
            h_flip_flag = randint(0, 1)
            v_flip_flag = randint(0, 1)
            if h_flip_flag == 1:
                im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
                im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
                im3 = im3.transpose(Image.FLIP_LEFT_RIGHT)
            if v_flip_flag == 1:
                im1 = im1.transpose(Image.FLIP_TOP_BOTTOM)
                im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)
                im3 = im3.transpose(Image.FLIP_TOP_BOTTOM)

            order_reverse = randint(0, 1)
            if order_reverse == 1:
                tmp = im1
                im1 = im3
                im3 = tmp

        if self.crop_size is not None:
            # random crop
            frame_w, frame_h = im1.size
            if self.is_train:
                crop_from_H = randint(0, frame_h - self.crop_size[1])
                crop_from_W = randint(0, frame_w - self.crop_size[0])
            else:
                crop_from_H, crop_from_W = 0, 0
            im1 = im1.crop((crop_from_W, crop_from_H, crop_from_W + self.crop_size[0], crop_from_H + self.crop_size[1]))
            im2 = im2.crop((crop_from_W, crop_from_H, crop_from_W + self.crop_size[0], crop_from_H + self.crop_size[1]))
            im3 = im3.crop((crop_from_W, crop_from_H, crop_from_W + self.crop_size[0], crop_from_H + self.crop_size[1]))

            # random rotate.
            if self.is_train and self.crop_size[0] == self.crop_size[1]:
                angle = randint(0, 3)
                im1 = im1.rotate(90 * angle)
                im2 = im2.rotate(90 * angle)
                im3 = im3.rotate(90 * angle)

        im1 = to_tensor(im1)
        im2 = to_tensor(im2)
        im3 = to_tensor(im3)

        return torch.stack([im1, im3], dim=1), im2, 0.5, name


# modified from:
# https://github.com/myungsub/CAIN/blob/master/data/snufilm.py
class SNU_FILM(data.Dataset):
    def __init__(self, data_root='/dataset/SNU_FILM', mode='hard', crop=None):
        super().__init__()
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        self.data_root = data_root
        test_fn = os.path.join(data_root, 'eval_modes', f'test-{mode}.txt')
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]
        self.crop = crop if crop is None or isinstance(crop, tuple) else (crop, crop)
        
    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        imgpaths = self.frame_list[idx]
        
        # customization
        # rename datapath to my path.
        for i, imgpath in enumerate(imgpaths):
            imgpaths[i] = imgpath.replace('data/SNU-FILM', self.data_root)
        
        img1 = Image.open(os.path.join(self.data_root, imgpaths[0])).convert('RGB')
        img2 = Image.open(os.path.join(self.data_root, imgpaths[1])).convert('RGB')
        img3 = Image.open(os.path.join(self.data_root, imgpaths[2])).convert('RGB')

        # customization
        # cropping
        if self.crop is not None:
            img1 = img1.crop((0, 0, self.crop[1], self.crop[0]))
            img2 = img1.crop((0, 0, self.crop[1], self.crop[0]))
            img3 = img1.crop((0, 0, self.crop[1], self.crop[0]))

        img1 = to_tensor(img1)
        img2 = to_tensor(img2)
        img3 = to_tensor(img3)
        name = '/'.join(imgpaths[1].split('/')[-3:])

        return torch.stack([img1, img3], dim=1), img2, 0.5, name


# referenced LDMVFI repo.
class Middlebury_others(data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.input_path = os.path.join(path, 'other-color-allframes/other-data')
        self.gt_path = os.path.join(path, 'other-gt-interp')
        self.seqs = sorted(glob(os.path.join(self.gt_path, '*')))

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        video = self.seqs[idx]
        seq_name = video.split('/')[-1]
        gt = to_tensor(Image.open(os.path.join(video, 'frame10i11.png')).convert('RGB'))
        img1 = to_tensor(Image.open(os.path.join(self.input_path, seq_name, 'frame10.png')).convert('RGB'))
        img3 = to_tensor(Image.open(os.path.join(self.input_path, seq_name, 'frame11.png')).convert('RGB'))

        return torch.stack([img1, img3], dim=1), gt, 0.5, seq_name


# modified from softsplat repo.
class Xiph(data.Dataset):
    def __init__(self, path, setting='4K', extract_frames=True, from_link=False):
        super().__init__()
        self.path = os.path.join(path, 'frames')
        self.setting = setting
        os.makedirs(self.path, exist_ok=True)

        if extract_frames:
            if len(glob(f'{self.path}/BoxingPractice-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/BoxingPractice-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/BoxingPractice-%03d.png')
            # end
            if len(glob(f'{self.path}/Crosswalk-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/Crosswalk-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/Crosswalk-%03d.png')
            # end
            if len(glob(f'{self.path}/DrivingPOV-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/DrivingPOV-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/DrivingPOV-%03d.png')
            # end
            if len(glob(f'{self.path}/FoodMarket-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/FoodMarket-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/FoodMarket-%03d.png')
            # end
            if len(glob(f'{self.path}/FoodMarket2-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/FoodMarket2-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/FoodMarket2-%03d.png')
            # end
            if len(glob(f'{self.path}/RitualDance-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/RitualDance-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/RitualDance-%03d.png')
            # end
            if len(glob(f'{self.path}/SquareAndTimelapse-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/SquareAndTimelapse-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/SquareAndTimelapse-%03d.png')
            # end
            if len(glob(f'{self.path}/Tango-*.png')) != 100:
                if from_link:
                    os.system(f'ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/Tango-%03d.png')
                else:
                    os.system(f'ffmpeg -i {path}/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 {self.path}/Tango-%03d.png')
            # end
        
        # make triplets
        self.triplets = []
        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
            for intFrame in range(2, 99, 2):
                npyOne = f'{self.path}/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png'
                npyTwo = f'{self.path}/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png'
                npyTruth = f'{self.path}/' + strFile + '-' + str(intFrame).zfill(3) + '.png'
                triplet = (npyOne, npyTruth, npyTwo)
                self.triplets.append(triplet)

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        npyOne = cv2.imread(triplet[0], flags=-1)
        npyTruth = cv2.imread(triplet[1], flags=-1)
        npyTwo = cv2.imread(triplet[2], flags=-1)

        if self.setting == '2K':  # resized
            npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
        elif self.setting == '4K':  # cropped
            npyOne = npyOne[540:-540, 1024:-1024, :]
            npyTwo = npyTwo[540:-540, 1024:-1024, :]
            npyTruth = npyTruth[540:-540, 1024:-1024, :]

        npyOne = cv2.cvtColor(npyOne, cv2.COLOR_BGR2RGB)
        npyTruth = cv2.cvtColor(npyTruth, cv2.COLOR_BGR2RGB)
        npyTwo = cv2.cvtColor(npyTwo, cv2.COLOR_BGR2RGB)

        tenOne = torch.FloatTensor(np.ascontiguousarray(npyOne.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(np.ascontiguousarray(npyTwo.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        tenGT = torch.FloatTensor(np.ascontiguousarray(npyTruth.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        name = '/'.join(triplet[1].split('/')[-2:])
        
        return torch.stack([tenOne, tenTwo], dim=1), tenGT, 0.5, name


# for demo, taking video inputs
class VideoData(data.Dataset):
    def __init__(self, video, save_as_png=False):
        super().__init__()
        buffer_path = None
        if save_as_png:
            buffer_path = f'./frames_buffer'
        self.frames_list = video2frames(video, output_path=buffer_path)
        self.save_as_png = save_as_png

    def __len__(self):
        return len(self.frames_list) - 1
    
    def __getitem__(self, idx):
        frame0 = self.frames_list[idx]
        frame1 = self.frames_list[idx + 1]
        if self.save_as_png:
            frame0 = cv2.imread(frame0, cv2.IMREAD_COLOR)
            frame1 = cv2.imread(frame1, cv2.IMREAD_COLOR)
        frame0 = to_tensor(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
        frame1 = to_tensor(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        return frame0, frame1