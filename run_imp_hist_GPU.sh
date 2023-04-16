#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0
nvcc img_imp_hist_GPU.cu -O2 -o image_imp_hist
#srun --reservation=fri -G1 -n1 cuda-memcheck image_hist lena.png
srun --reservation=fri -G1 -n1 image_imp_hist lena.png
