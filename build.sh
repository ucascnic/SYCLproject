export MKLROOT=/home/gta/brianshi/builds/__release_lnx/mkl
export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}
source /home/gta/compilers/20230129_rls/lnx/setvars.sh > /dev/null 2>&1

# Compile with MKL + DPCPP
icpx -fsycl -w -x c++ -DMKL_ILP64 test.mod.cu -DEXCLUDE_BLAS=0 -DBUILD_FOR_MKL=1 -I./include -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o test_mkl_dpcpp

# Compile with DPCPP and no MKL
icpx -fsycl -w -x c++ -I./include test.mod.cu -DEXCLUDE_BLAS=1 -DBUILD_FOR_MKL=1 -o test_dpcpp

# Compile with CUDA + cuBLAS
# nvcc -DEXCLUDE_BLAS=0 -DBUILD_FOR_MKL=0 test.mod.cu -lcublas test_cuda

# Compile with CUDA and no cuBLAS
# nvcc -DEXCLUDE_BLAS=1 -DBUILD_FOR_MKL=0 test.mod.cu test_cuda
