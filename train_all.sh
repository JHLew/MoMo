# root directory of datasets: ${1}
# initial synthesis network training with L1 loss
accelerate launch train_synth.py --name momo_synth --dataroot ${1} --n_epochs 150 --metric psnr --w_lpips 0 --w_style 0
# tuning synthesis network with Ls loss
accelerate launch train_synth.py --name momo_synth --dataroot ${1} --resume 150 --n_epochs 200 --metric lpips --w_lpips 1 --w_style 20

# flow teacher tuning
accelerate launch train_flow_teacher.py --name momo_teacher_Ls --dataroot ${1} --n_epochs 100 --synth_model momo_synth --metric lpips --w_lpips 1 --w_style 20

# Motion Modeling with diffusion
accelerate launch train_diffusion.py --name momo_full --dataroot ${1} --teacher momo_teacher_Ls --mp no --visualize_flows