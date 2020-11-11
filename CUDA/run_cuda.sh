nvcc -std=c++11 -arch=compute_52 -code=sm_52 main.cu cuda_smith_waterman_skeleton.cu -o cuda_smith_waterman
num_block=4
num_thread=32
# input=./datasets/input3.txt
# input=./datasets/input5.txt
input=./datasets/4k.in
# input=./datasets/input6.txt
echo $input $num_block $num_thread
./cuda_smith_waterman $input $num_block $num_thread
