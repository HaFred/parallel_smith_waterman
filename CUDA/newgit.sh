nvcc -std=c++11 -arch=compute_52 -code=sm_52 -I "/usr/local/cuda-8.0.61/include" cuda_sw_ori.cu -o cuda_smith_waterman_git
./cuda_smith_waterman_git 