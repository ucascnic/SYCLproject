

 
nvcc -DEXCLUDE_BLAS=0 -DBUILD_FOR_MKL=0 test.mod.cu -lcublas -o test_cuda
srun -N 1  ./test_cuda 1.txt