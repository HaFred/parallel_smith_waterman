nvcc -std=c++11 -arch=compute_52 -code=sm_52 main.cu cuda_smith_waterman_skeleton_exp_shr.cu -o cuda_smith_waterman
num_block=2
num_thread=32
# input=./datasets/sample2.in
input=./datasets/sample.in
# input=./datasets/1k.in
# input=./datasets/4k.in
# input=./datasets/input6.txt
echo $input $num_block $num_thread
./cuda_smith_waterman $input $num_block $num_thread
