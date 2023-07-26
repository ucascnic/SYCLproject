# SYCLproject
SYCLproject. Learning how to write and optimize the sycl code.


# 编译命令为
 
##CUDA

'''nvcc admmsolver.cu  cnic_sparsematrix.cu -lcusparse -lcublas'''

##sycl
'''dpcpp admmsolver.dp.cpp cnic_sparsematrix.dp.cpp cnicsparsematrix.cpp -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl   -fsycl -DMKL_ILP64 -qmkl=parallel -DPSTL_USE_PARALLEL_POLICIES=0'''