# Disentangled Motion Modeling for Video Frame Interpolation
<div align="center">
  <img src=figures/cover_figure.png width=900 />
</div>



## Environmental settings
```
conda create -n momo python=3.10.9
conda activate momo
pip install -r requirements.txt
```

To configure specific GPUs to use in training / inference, use the `accelerate config` command before running the commands below.

### Datasets
We use [Vimeo90k](http://toflow.csail.mit.edu/) for training, and use [SNU-FILM](https://myungsub.github.io/CAIN/), [Xiph](dataset.py#L168), [Middlebury-others](https://vision.middlebury.edu/flow/data/) for validation. Download the datasets and put them under a root directory of datasets (e.g., /dataset).

The datasets should have a directory structure as follows:
```
└─ <dataroot>
    ├─ vimeo_triplet
    |   ├─ sequences/
    |   ├─ tri_trainlist.txt
    |   ├─ tri_testlist.txt
    |   └─ readme.txt
    |
    ├─ SNU_FILM
    |   ├─ eval_modes
    |   |   ├─ test-easy.txt
    |   |   ├─ test-medium.txt
    |   |   ├─ test-hard.txt
    |   |   ├─ test-extreme.txt
    |   └─ test
    |       ├─ GOPRO_test/
    |       └─ YouTube_test/
    |
    ├─ Middlebury
    |   └─ other-gt-interp
    |       ├─ Beanbags/
    |       ├─ DogDance/
    |       ...
    |       ├─ Urban3/
    |       └─ Walking/
    |
    └─ Xiph (could be downloaded / created with dataset.py)
        └─ frames
            ├─ BoxingPractice-001.png
            ├─ BoxingPractice-002.png
            ...
            ├─ Tango-099.png
            └─ Tango-100.png
```


## Quick start
### Download the pretrained model
The pretrained weights of our model can be downloaded from [here](https://drive.google.com/drive/folders/1k-7JW9krOgzDjzsEKDgS7bgvU1pX79nT?usp=sharing).
The `diffusion.zip` file corrsponds to the entire MoMo framework. (The files `flow_teacher.zip` and `synthesis.zip` corresponds to the flow teacher model and the synthesis network, respectively.)

Download the zip files and extract them each under `experiments` folder.
It should then have a hierarchy like: `experiments/diffusion/momo_full/weights/model.pth`.

### Running x2 interpolation of a video
For x2 interpolation of a video, simply run the command below
```
accelerate launch demo.py --video <path_to_video.mp4> --output_path <path_to_x2_video.mp4>
```
Note: you can also use `python` command instead of `accelerate` for single GPU inference.
```
python demo.py --video <path_to_video.mp4> --output_path <path_to_x2_video.mp4>
```

### Interpolation result
Below is an example x2 interpolation result with our model:

#### Input Video
<div align="center">
  <img src=figures/Xiph4K_SquareAndTimelapse_input.gif width=600 />
</div>

#### Output Video
<div align="center">
  <img src=figures/Xiph4K_SquareAndTimelapse_x2_result.gif width=600 />
</div>

## Training

The training process of our framework consist of several steps.
You can run the following script file for training, and details can be found in `train_all.sh`.
You may want to configure the path to the root directory where datasets are placed.
```
sh train_all.sh <path_to_dataroot>
```
<div align="center">
  <img src=figures/overview.png width=700 />
</div>

## Testing


### Test full model (MoMo)
For evaluation on public benchmarks, run the command as below.
You can optionally choose to visualize the estimated flows by `--visualize_flows`. You can save the results either as tensorboard logs by `--logging`, or as png by `--save_as_png`. When saving the results as png files, make sure to set the directory to save the results by `--png_save_dir`.
```
# basic usage
accelerate launch eval_momo.py --name momo_full --dataroot <path_to_dataroot> --valid_dataset <dataset_name>

# example usage of testing on SNU-FILM-hard, log results on tensorboard
accelerate launch eval_momo.py --name momo_full --dataroot <path_to_dataroot> --visualize_flows --valid_dataset SNU_FILM_hard --logging

# example usage of testing on Xiph-2K data, visualize the estimated flow maps and save all results as png
accelerate launch eval_momo.py --name momo_full --dataroot <path_to_dataroot> --valid_dataset Xiph_2K --visualize_flows --save_as_png --png_save_dir ./momo_png_results/Xiph_2K
```

### Test individual components (i.e., synthesis model, teacher flow model)
The individual components can be tested with the command below.
The weights of the synthesis model, and teacher flow model are also available [here](https://drive.google.com/drive/folders/1k-7JW9krOgzDjzsEKDgS7bgvU1pX79nT?usp=sharing).
```
# synthesis model testing
accelerate launch eval_components.py --name momo_synth --dataroot <path_to_dataroot> --component synthesis

# teacher flow model testing
accelerate launch eval_components.py --name momo_teacher_Ls --dataroot <path_to_dataroot> --component flow_teacher
```

## Citation
