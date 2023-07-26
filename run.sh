export MKLROOT=/home/gta/brianshi/builds/__release_lnx/mkl
export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}
source /home/gta/compilers/20221109_rls/lnx/setvars.sh > /dev/null 2>&1

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 

ZE_AFFINITY_MASK=0.0 SYCL_DEVICE_FILTER=level_zero:gpu ./test_dpcpp
#ZE_AFFINITY_MASK=0.0 SYCL_DEVICE_FILTER=level_zero:gpu ./test_mkl_dpcpp
