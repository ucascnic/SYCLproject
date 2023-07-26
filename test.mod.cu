#ifndef BUILD_FOR_MKL
#define BUILD_FOR_MKL 1
#endif

#ifndef EXCLUDE_BLAS
#define EXCLUDE_BLAS  0
#endif

#if BUILD_FOR_MKL
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <dpct/dpl_utils.hpp>
#if EXCLUDE_BLAS
namespace oneapi::mkl::blas::column_major {
    template <typename T>
    sycl::event nrm2(sycl::queue &q, int64_t n, T *x, int64_t incx, T *res) {
        return sycl::event();
    };

    template <typename T>
    sycl::event dot(sycl::queue &q, int64_t n, T *x, int64_t incx, T *y, int64_t incy, T *res) {
        return sycl::event();
    };
}
#else
#include <oneapi/mkl.hpp>
#endif
#else // BUILD_FOR_CUBLAS
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include<cuda_runtime_api.h>
#if EXCLUDE_BLAS

typedef int cublasStatus_t;
typedef int cublasHandle_t;

cublasStatus_t cublasDdot(cublasHandle_t handle, int n,
                          const double          *x, int incx,
                          const double          *y, int incy,
                          double          *result) {
    return 0;
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n,
                           const double          *x, int incx,
                           double          *result) {
    return 0;
}

cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return 0;
}

#else
#include<cublas_v2.h>
#endif

#endif

#include <chrono>
/*By Auber #define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
using namespace arma;*/

std::chrono::duration<double> dot_time(0);
std::chrono::duration<double> dot_submit_time(0);
int64_t dot_count = 0;
int64_t dot_ignore = 0;
std::chrono::duration<double> nrm2_time(0);
int64_t nrm2_count = 0;
int64_t nrm2_ignore = 0;

std::chrono::duration<double> ker1_time(0);
int64_t ker1_count = 0;
int64_t ker1_ignore = 0;
std::chrono::duration<double> ker2_time(0);
int64_t ker2_count = 0;
int64_t ker2_ignore = 0;
std::chrono::duration<double> ker3_time(0);
int64_t ker3_count = 0;
int64_t ker3_ignore = 0;

std::chrono::duration<double> mem1_time(0);
int64_t mem1_count = 0;
int64_t mem1_ignore = 0;
std::chrono::duration<double> mem2_time(0);
int64_t mem2_count = 0;
int64_t mem2_ignore = 0;

template <typename T>
class Resources
{
public:
    int max_resources;
    int cnt;
    T *resources;
public:

    Resources(int n){
        this->max_resources = n;
        this->cnt = 0;

        T *temp = NULL;
#if BUILD_FOR_MKL
        temp = (T *) sycl::malloc_host(sizeof(T) * n, dpct::get_default_queue());
#else
        temp = (T *) malloc(sizeof(T) * n);
#endif

        for (int i = 0 ; i < n; i++) {
            double f = (double)rand() / RAND_MAX;
            temp[i] = -0.00025 + f * (0.0005);
        }

#if BUILD_FOR_MKL
        // this can't be used in lambda.
        T *resources_d = (T*) sycl::malloc_device(sizeof(T) * this->max_resources,
                                            dpct::get_default_queue());

        // For some reason memcpy segfaults..
        dpct::get_default_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range(n), [=](sycl::item<1> idx) {
                resources_d[idx] = temp[idx];
            });
        }).wait();
        resources = resources_d;
        sycl::free(temp, dpct::get_default_queue());
#else
        cudaMalloc((void**)&this->resources,sizeof(T) * this->max_resources);
        cudaMemcpy(this->resources,temp,sizeof(T)*n,cudaMemcpyHostToDevice);
        free(temp);
#endif

    }
    ~Resources(){
#if BUILD_FOR_MKL
        sycl::free(this->resources, dpct::get_default_queue());
#else
        cudaFree(this->resources);
#endif
    }

    T* allocate_resource(int required){
        int nn = this->cnt;
        this->cnt += required;
        if (this->cnt  > this->max_resources){
            printf("do not have so much resources");
            exit(0);
        }

        return  &this->resources[nn];
    }

};

struct CSRIntMatrix{
    std::vector<ptrdiff_t> ptr, col;
    std::vector<int> val;
};

struct COORowPtrl{
   std::vector<ptrdiff_t> ptr;
};

template <typename T>
void show_res_T(T *s,int n){
    T *res = (T *)malloc(n*sizeof(T));
#if BUILD_FOR_MKL
    dpct::get_default_queue().memcpy(res, s, n * sizeof(T)).wait();
#else
    cudaMemcpy(res,s,n*sizeof(T),cudaMemcpyDeviceToHost);
#endif

    for (int i = 0 ; i< n;++i){
        printf("%.8f\t",res[i]);
    }

    printf("\n");
    free(res);
}

class CSRMatrix
{
public:
    CSRMatrix(int size,int rows,int cols);
    //By Auber void print_matrix();
public:
#if BUILD_FOR_MKL
    dpct::device_vector<double> cudata;
    dpct::device_vector<int> csr_row_ptrl;
    dpct::device_vector<int> cucol;
#else
    thrust::device_vector<double> cudata;
    thrust::device_vector<int> csr_row_ptrl;
    thrust::device_vector<int> cucol;
#endif


    int  n_element;
    int n_rows;
    int n_cols;

};

CSRMatrix::CSRMatrix(int size,int rows,int cols){
    this->n_element = size;
    this->n_rows = rows;
    this->n_cols = cols;
}


#if BUILD_FOR_MKL
void update_rew(double *rnew, double *b, int nn, sycl::nd_item<3> item_ct1) {
    // xNew = xOld  + alpha.*dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    rnew[x] = b[x] - rnew[x];


}

void updateXnew(double *xOld, double *dOld, double *xNew, double alpha, int nn,
                sycl::nd_item<3> item_ct1) {
    // xNew = xOld  + alpha.*dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    xNew[x] = xOld[x] + alpha * dOld[x];
    xOld[x] = xNew[x];


}

void updatedOld(double *rNew, double *dOld, double alpha, int nn,
                sycl::nd_item<3> item_ct1) {
    //dOld = rNew + alpha* dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];

}

#else
__global__ void update_rew(double* rnew,double *b, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    rnew[x] = b[x] - rnew[x];


}

__global__ void updateXnew(double* xOld, double*dOld,double*xNew, double alpha, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    xNew[x] = xOld[x] + alpha * dOld[x];
    xOld[x] = xNew[x];


}

__global__ void updatedOld(double *rNew,double *dOld,double alpha, int nn){
    //dOld = rNew + alpha* dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];

}
#endif

void sp_matrix_times_V_ptrl(CSRMatrix*A,double*b,double *c) { }

/* DPCT_ORIG void Conjugate_gradient_sp(cublasHandle_t handle,CSRMatrix
 * *A,double *b, double *x0,*/
template<bool BenchTime = false> // true = time MKL functions, false = don't time
#if BUILD_FOR_MKL
void Conjugate_gradient_sp_mod(sycl::queue *handle, CSRMatrix *A, double *b,
                           double *x0, double tol, int itMax, double *recources,
                           int n) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
#else
void Conjugate_gradient_sp_mod(cublasHandle_t &handle,CSRMatrix *A,double *b, double *x0,
                           double tol,int itMax,double *recources,int n){
#endif
    std::chrono::time_point<std::chrono::steady_clock> start,end,mid;

    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
#if BUILD_FOR_MKL
    sycl::range<3> block(1, 1, (n) / 128 + 1);
#else
    dim3 block( (n)/128+ 1);
#endif
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
#if BUILD_FOR_MKL
    q_ct1.memcpy(xOld, x0, size_x).wait();
#else
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);
#endif

    sp_matrix_times_V_ptrl(A,xOld,rOld);

#if BUILD_FOR_MKL
    q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           update_rew(rOld, b, n, item_ct1);
                       }).wait();
    q_ct1.memcpy(dOld, rOld, size_x).wait();
#else
    update_rew<<<block,128>>>(rOld,b,n);
    cudaDeviceSynchronize();
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);
#endif

#if BUILD_FOR_MKL
    double *bNorm = sycl::malloc_device<double>(1, dpct::get_default_queue());
    oneapi::mkl::blas::column_major::nrm2(*handle, n, b, 1, bNorm).wait(); //bNorm;
#else
    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;
#endif

#if BUILD_FOR_MKL
    double *alpha_ = sycl::malloc_device<double>(1, dpct::get_default_queue());
    double *beta_ = sycl::malloc_device<double>(1, dpct::get_default_queue());
    double *alpha_temp = sycl::malloc_device<double>(1, dpct::get_default_queue());
    double *resNorm = sycl::malloc_device<double>(1, dpct::get_default_queue());

    dpct::get_default_queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range(1), [=](sycl::item<1> idx) {
            alpha_[idx] = 1.0;
            beta_[idx] = 1.0;
            alpha_temp[idx] = 1.0;
            resNorm[idx] = 1.0;
        });
    }).wait();
    sycl::event evt;
#else
    double alpha_ = 1.0;
    double beta_ = 1.0;
    double alpha_temp = 1.0;
    double resNorm = 1.0;
#endif


    for (int k =0; k <= itMax; ++k){
        //alpha = (dOld'*rOld)/(dOld'*A*dOld);

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt = oneapi::mkl::blas::column_major::dot(*handle, n, dOld, 1, rOld, 1, alpha_temp);
#else
        cublasDdot(handle,n,dOld,1,rOld,1,&alpha_temp);
#endif
        if constexpr (BenchTime) mid = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt.wait();
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) dot_count++;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_time += end - start;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_submit_time += mid - start;

        sp_matrix_times_V_ptrl(A,dOld,local_temp); // Does nothing..

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt = oneapi::mkl::blas::column_major::dot(*handle, n, local_temp, 1, dOld, 1, alpha_);
#else
        cublasDdot(handle,n,local_temp,1,dOld,1,&alpha_);
#endif
        if constexpr (BenchTime) mid = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt.wait();
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) dot_count++;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_time += end - start;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_submit_time += mid - start;

        // xNew = xOld  + alpha.*dOld;
/* DPCT_ORIG         */
        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {

                               auto alpha__ = alpha_temp[0]/(alpha_[0]);

                               updateXnew(xOld, dOld, xNew, alpha__, n, item_ct1);
                           }).wait();
#else
        alpha_ = alpha_temp/alpha_;
        updateXnew<<<block,128>>>(xOld,dOld,xNew,alpha_,n);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) ker1_count++;
        if constexpr (BenchTime) if(ker1_count > ker1_ignore) ker1_time += end - start;
        //rNew = b-A*xNew;

        sp_matrix_times_V_ptrl(A,xNew,rNew); // Does nothing..

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               update_rew(rNew, b, n, item_ct1);
                           }).wait();
#else
        update_rew<<<block,128>>>(rNew,b,n);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) ker2_count++;
        if constexpr (BenchTime) if(ker2_count > ker2_ignore) ker2_time += end - start;

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        oneapi::mkl::blas::column_major::nrm2(*handle, n, rNew, 1, resNorm).wait();
#else
        cublasDnrm2(handle,n,rNew,1,&resNorm);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) nrm2_count++;
        if constexpr (BenchTime) if(nrm2_count > nrm2_ignore) nrm2_time += end - start;

        /* SKIP tolerance check..
        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
        handle->memcpy(resNorm_, res_temp_ptr_ct2, sizeof(double)).wait(); // device -> host copy, expensive

        if (*resNorm_ < tol * bNorm){
#if BUILD_FOR_MKL
            q_ct1.memcpy(x0, xNew, size_x).wait();
#else
            cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);
#endif
            return ;
        }
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) mem1_count++;
        if constexpr (BenchTime) if(mem1_count > mem1_ignore) mem1_time += end - start;*/

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        oneapi::mkl::blas::column_major::dot(*handle, n, rNew, 1, rNew, 1, alpha_temp);
#else
        cublasDdot(handle,n,rNew,1,rNew,1,&alpha_temp);
#endif
        if constexpr (BenchTime) mid = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt.wait();
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) dot_count++;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_time += end - start;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_submit_time += mid - start;

        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt = oneapi::mkl::blas::column_major::dot(*handle, n, rOld, 1, rOld, 1, alpha_);
#else
        cublasDdot(handle,n,rOld,1,rOld,1,&alpha_);
#endif
        if constexpr (BenchTime) mid = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        evt.wait();
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) dot_count++;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_time += end - start;
        if constexpr (BenchTime) if(dot_count > dot_ignore) dot_submit_time += mid - start;

        //dOld = rNew + beta* dOld;
        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {

                               auto beta__ = alpha_temp[0]/alpha_[0];

                               updatedOld(rNew, dOld, beta__, n, item_ct1);
                           }).wait();
#else
        beta_ = alpha_temp/alpha_;
        updatedOld<<<block,128>>>(rNew,dOld,beta_,n);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) ker3_count++;
        if constexpr (BenchTime) if(ker3_count > ker3_ignore) ker3_time += end - start;
        //rOld = rNew;
        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        q_ct1.memcpy(rOld, rNew, size_x).wait();
#else
        cudaMemcpy(rOld,rNew,size_x,cudaMemcpyDeviceToDevice);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) mem1_count++;
        if constexpr (BenchTime) if(mem1_count > mem1_ignore) mem1_time += end - start;
        //xOld = xNew;
        if constexpr (BenchTime) start = std::chrono::steady_clock::now();
#if BUILD_FOR_MKL
        q_ct1.memcpy(xOld, xNew, size_x).wait();
        q_ct1.wait();
#else
        cudaMemcpy(xOld,xNew,size_x,cudaMemcpyDeviceToDevice);
#endif
        if constexpr (BenchTime) end = std::chrono::steady_clock::now();
        if constexpr (BenchTime) mem2_count++;
        if constexpr (BenchTime) if(mem2_count > mem2_ignore) mem2_time += end - start;
    }
#if BUILD_FOR_MKL
    q_ct1.memcpy(x0, xNew, size_x).wait();
#else
    cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);
#endif

#if BUILD_FOR_MKL
    sycl::free(bNorm, dpct::get_default_queue());
    sycl::free(alpha_, dpct::get_default_queue());
    sycl::free(beta_, dpct::get_default_queue());
    sycl::free(alpha_temp, dpct::get_default_queue());
    sycl::free(resNorm, dpct::get_default_queue());
#endif

}

int main () {
    srand(0);
#if BUILD_FOR_MKL
    sycl::queue *handle;
    handle = &dpct::get_default_queue();
#else
    cublasHandle_t handle;
    cublasCreate(&handle);
#endif

    //COOMatrix   AAA_coo(4, 4, 16);
    //CSRMatrix   AAA_cu( AAA_coo.n_element,AAA_coo.n_rows,AAA_coo.n_cols);
    //AAA_cu.create_matrix_from_coo(&AAA_coo);

    CSRMatrix   AAA_cu( 16,4,4);

    Resources<double>  resources(1<<23);
    int equation_size = 1024;//By Auber bbsize+beqy.n_elem;
    double * bbb = resources.allocate_resource(equation_size);
    double * yy = resources.allocate_resource(equation_size);
    double * buff = resources.allocate_resource(8*equation_size);

    constexpr bool BenchTime = true;//false; // Get timing for MKL/cuBLAS dot/nrm2/other kernels

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 10; iter++){
        Conjugate_gradient_sp_mod<BenchTime>(handle,&AAA_cu,bbb,yy,1e-1/((double)iter),1e3,buff,equation_size);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time interval : " << diff.count() << " s\n";

    // Extra debug info..
    if (BenchTime) {
#if !EXCLUDE_BLAS
        std::cout << "Total dot time        : " << dot_time.count() << "s\n";
        std::cout << "Total nrm2 time       : " << nrm2_time.count() << "s\n";
#endif
        std::cout << "Total updateXnew time : " << ker1_time.count() << "s\n";
        std::cout << "Total update_rew time : " << ker2_time.count() << "s\n";
        std::cout << "Total updateOld time  : " << ker3_time.count() << "s\n";
        std::cout << "Total memcpy1 time    : " << mem1_time.count() << "s\n";
        std::cout << "Total memcpy2 time    : " << mem2_time.count() << "s\n";
#if !EXCLUDE_BLAS
        printf("Avg dot time          : %lfs, %ld calls\n",dot_time.count()/((double)(dot_count - dot_ignore)), dot_count);
#if BUILD_FOR_MKL
        // For cuBLAS calls are blocking due to alpha/beta scalars
        printf("Avg dot submit time   : %lfs, %ld calls\n",
               dot_submit_time.count()/((double)(dot_count - dot_ignore)), dot_count);
#endif
        printf("Avg nrm2 time         : %lfs, %ld calls\n",nrm2_time.count()/((double)(nrm2_count - nrm2_ignore)), nrm2_count);
#endif
        printf("Avg updateXnew time   : %lfs, %ld calls\n",ker1_time.count()/((double)(ker1_count - ker1_ignore)), ker1_count);
        printf("Avg update_rew time   : %lfs, %ld calls\n",ker2_time.count()/((double)(ker2_count - ker2_ignore)), ker2_count);
        printf("Avg updateOld time    : %lfs, %ld calls\n",ker3_time.count()/((double)(ker3_count - ker3_ignore)), ker3_count);
        printf("Avg memcpy1 time      : %lfs, %ld calls\n",mem1_time.count()/((double)(mem1_count - mem1_ignore)), mem1_count);
        printf("Avg memcpy2 time      : %lfs, %ld calls\n",mem2_time.count()/((double)(mem2_count - mem2_ignore)), mem2_count);
    }

    printf("\nPrinting Result: \n");
    show_res_T(yy, 10);

  return 0;
}
