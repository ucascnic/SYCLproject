# SYCLproject
SYCLproject. Learning how to write and optimize the sycl code.



运行示例
'''
 ./test_mkl_dpcpp 1.txt
 5 5
Time interval : 1.85923 s
Total dot time        : 0.460202s
Total nrm2 time       : 0.156654s
Total updateXnew time : 0.395826s
Total update_rew time : 0.231144s
Total updateOld time  : 0.33895s
Total memcpy1 time    : 0.107572s
Total memcpy2 time    : 0.156954s
Avg dot time          : 0.000011s, 40040 calls
Avg dot submit time   : 0.000006s, 40040 calls
Avg nrm2 time         : 0.000016s, 10010 calls
Avg updateXnew time   : 0.000040s, 10010 calls
Avg update_rew time   : 0.000023s, 10010 calls
Avg updateOld time    : 0.000034s, 10010 calls
Avg memcpy1 time      : 0.000011s, 10010 calls
Avg memcpy2 time      : 0.000016s, 10010 calls

Printing Result:
3701.77853618   -1147.28902069  4218.15361148   2674.87496941   5088.62588762   3701.77853618 -1147.28902069   4218.15361148   2674.87496941   5088.62588762


'''



'''
srun -N 1  ./test_cuda 1.txt
 5 5
Time interval : 1.29805 s
Total dot time        : 0.736239s
Total nrm2 time       : 0.391209s
Total updateXnew time : 0.0274747s
Total update_rew time : 0.027705s
Total updateOld time  : 0.0281982s
Total memcpy1 time    : 0.0385969s
Total memcpy2 time    : 0.0360667s
Avg dot time          : 0.000018s, 40040 calls
Avg nrm2 time         : 0.000039s, 10010 calls
Avg updateXnew time   : 0.000003s, 10010 calls
Avg update_rew time   : 0.000003s, 10010 calls
Avg updateOld time    : 0.000003s, 10010 calls
Avg memcpy1 time      : 0.000004s, 10010 calls
Avg memcpy2 time      : 0.000004s, 10010 calls

Printing Result: 
3701.77853618   -1147.28902069  4218.15361148   2674.87496941   5088.62588762   3701.77853618   -1147.28902069  4218.15361148       2674.87496941   5088.62588762
'''
