nvcc -std=c++11 -arch=compute_52 -code=sm_52 cuda_sw_ori.cu -o cuda_smith_waterman_git
num_block=4
num_thread=32
input=./datasets/1k.in
echo $input $num_block $num_thread
./cuda_smith_waterman_git $input $num_block $num_thread
