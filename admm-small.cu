
#include<vector>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include<cuda_runtime_api.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include<cublas_v2.h>
#include<fstream>
#include <random>
#include <chrono>

class COOMatrix
{
public:
    COOMatrix(){};
    COOMatrix(int m, int n,int n_ele);
    

    void read_fromfile(std::string& f);


    void to_GPU();
    void print_matrix();
    int n_element;
    int n_rows;
    int n_cols;


public:

    thrust::host_vector<double> data;
    thrust::host_vector<int> row;
    thrust::host_vector<int> col;

    thrust::device_vector<double> cudata;
    thrust::device_vector<int> curow;
    thrust::device_vector<int> cucol;

};


class CSRMatrix
{
public:
    void create_matrix_from_coo(COOMatrix * coo_matrix);
    CSRMatrix(int size,int rows,int cols);
    void print_matrix();
public:
    thrust::device_vector<double> cudata;
    thrust::device_vector<int> csr_row_ptrl;
    thrust::device_vector<int> cucol;
    int  n_element;
    int n_rows;
    int n_cols;

};

void COOMatrix::read_fromfile(std::string &f){


     
        std::ifstream file(f);
        int m, n, n_ele;
        file >> m >> n >> n_ele;
        data.resize(n_ele);
        row.resize(n_ele);
        col.resize(n_ele);
        for (int i = 0; i < n_ele; i++) {
            file >> row[i] >> col[i] >> data[i];
        }
        file.close();

    this->n_rows = m;
    this->n_cols = n;
    this->n_element = n_ele;
    this->cudata = this->data;
    this->curow = this->row;
    this->cucol = this->col;       
   
// 
}

COOMatrix::COOMatrix(int m, int n,int n_ele){


    this->n_cols = n;
    this->n_rows = m;
    this->n_element = n_ele;
    // cpudata

    this->data = std::vector<double>(n_ele,0.0);
    this->col = std::vector<int>(n_ele,0);
    this->row = std::vector<int>(n_ele,0);
    //gpu data
    this->cudata = this->data;
    this->curow = this->row;
    this->cucol = this->col;
}

void compressIndices(ptrdiff_t *idx, ptrdiff_t *idx_ptr, ptrdiff_t nindex, ptrdiff_t nelem)
{
    int curr_pos = 0, end_pos;
    idx_ptr[0] = 0;
    for (ptrdiff_t index = 0; index < nindex; index++)
    {
        for (end_pos = curr_pos; end_pos < nelem; end_pos++)
            if (idx[end_pos] > index) break;
        idx_ptr[index + 1] = end_pos;
        curr_pos = end_pos;
    }
    idx_ptr[nindex] = nelem;
}

struct CSRIntMatrix{
    std::vector<ptrdiff_t> ptr, col;
    std::vector<int> val;
};

struct COORowPtrl{
   std::vector<ptrdiff_t> ptr;
};

static int cmp_pair(ptrdiff_t M1, ptrdiff_t N1, ptrdiff_t M2, ptrdiff_t N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

template <typename T>
void qsortCOO2CSR(ptrdiff_t *row, ptrdiff_t *col, T *val, ptrdiff_t l, ptrdiff_t r)
{
    ptrdiff_t i = l, j = r, row_tmp, col_tmp;
    ptrdiff_t mid_row = row[(l + r) / 2];
    ptrdiff_t mid_col = col[(l + r) / 2];
    double val_tmp;
    while (i <= j)
    {
        while (cmp_pair(row[i], col[i], mid_row, mid_col)) i++;
        while (cmp_pair(mid_row, mid_col, row[j], col[j])) j--;
        if (i <= j)
        {
            row_tmp = row[i]; row[i] = row[j]; row[j] = row_tmp;
            col_tmp = col[i]; col[i] = col[j]; col[j] = col_tmp;
            val_tmp = val[i]; val[i] = val[j]; val[j] = val_tmp;

            i++;  j--;
        }
    }
    if (i < r) qsortCOO2CSR<T>(row, col, val, i, r);
    if (j > l) qsortCOO2CSR<T>(row, col, val, l, j);
}


void CSRMatrix::create_matrix_from_coo(COOMatrix * coo_matrix){

    int n = coo_matrix->n_element;
    this->cudata = coo_matrix->cudata;
    this->cucol = coo_matrix->cucol;
    this->csr_row_ptrl  = thrust::device_vector<int>(coo_matrix->n_rows + 1);

    COORowPtrl coorowptrl;
    coorowptrl.ptr.resize(n);
    for (int i =0;i<n;++i){
        coorowptrl.ptr[i] = coo_matrix->row[i];
    }

    CSRIntMatrix  csr;
    csr.col.resize(this->cucol.size());
    csr.ptr.resize(coo_matrix->n_rows+1);
    for (int i = 0 ;i < this->cucol.size();++i){
        csr.col[i]=coo_matrix->col[i];
    }


    ptrdiff_t nnz = n;
    csr.val.resize(nnz);
    for (int i = 0 ; i < nnz;++i){
        csr.val[i] = i;
    }


    qsortCOO2CSR<int>(coorowptrl.ptr.data(), csr.col.data(), csr.val.data(), 0, nnz - 1);

    compressIndices(coorowptrl.ptr.data(), csr.ptr.data(), coo_matrix->n_rows, nnz);


    thrust::host_vector<int> temp1(csr.ptr.size());

    for (int i = 0 ; i< csr.ptr.size();++i){
        temp1[i] = csr.ptr[i];

    }


    this->csr_row_ptrl = temp1;

    thrust::host_vector<double> temp2(csr.col.size());

    for (int i = 0 ; i< csr.col.size();++i){
        temp2[i] = csr.col[i];
    }
    this->cucol = temp2;


    thrust::host_vector<double> temp3 = coo_matrix->data;
    thrust::host_vector<double> temp4 = coo_matrix->data;
    for (int i = 0 ; i< temp3.size();++i){
        temp4[i] = temp3[csr.val[i]];
    }
    this->cudata = temp4;

    return   ;

}
CSRMatrix::CSRMatrix(int size,int rows,int cols){
    this->n_element = size;
    this->n_rows = rows;
    this->n_cols = cols;

}


__global__ void update_rew(double* rnew,double *b, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    rnew[x] = b[x] - rnew[x];


}

__global__ void updateXnew_(double* xOld, double*dOld,double*xNew, double *alpha, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    xNew[x] = xOld[x] + (*alpha) * dOld[x];
    xOld[x] = xNew[x];


}


__global__ void updateone(double* x, double*y,double*z){
    // z = x/y
    z[0]=x[0]/y[0];



}

#define WARP_ 32
inline __device__  double __shfl_down_(double var, unsigned int srcLane, int width=WARP_) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}

__global__ void kernal_mat_u_32_wrap(double *kernel_cudata,
                                     int *csr_row_ptrl,int *cucol,
                                     double *u_cudata,
                                     double *output_cudata,int n_rows){

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int vector_id = thread_id / WARP_;
    int lane_id = thread_id % WARP_;

    int row = vector_id;



    if(row <  n_rows){
        int begin_index =  csr_row_ptrl[row];
        int end_index =  csr_row_ptrl[row+1];

        double thread_sum = 0.0;
        for(int i = begin_index + lane_id; i < end_index; i+=WARP_)
            thread_sum += kernel_cudata[i] *  u_cudata [ cucol[i]];

//        if(row == 2423){
//            printf("csr_row_ptrl=%d   %d \n",csr_row_ptrl[row],csr_row_ptrl[row+1]);
//        }


        int temp = WARP_/2;
        while(temp >= 1){
            thread_sum += __shfl_down_(thread_sum, temp);
            temp >>= 1;
        }

        if ( lane_id == 0) {
              output_cudata[row] =  thread_sum;
        }

    }



}

void sp_matrix_times_V_ptrl(CSRMatrix*A,double*b,double *c){

    dim3 block(A->n_rows);
	

	

    kernal_mat_u_32_wrap<<<block,WARP_>>>(thrust::raw_pointer_cast(A->cudata.data()),
                                          thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
                                          thrust::raw_pointer_cast(A->cucol.data()),
                                          b,
                                          c,
                                          A->n_rows);


}

void Conjugate_gradient_sp(cublasHandle_t handle,CSRMatrix *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n){
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    dim3 block( (n)/128+ 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);

    sp_matrix_times_V_ptrl(A,xOld,rOld);

    update_rew<<<block,128>>>(rOld,b,n);


    // dOld  ;
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);

    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;

    //double alpha_ = 0.0;
    //double beta_ = 0.0;
    //double alpha_temp = 0.0;

    double resNorm = 0.0;

    double *alpha_ =  &recources[6*n];
    double *beta_ =  &recources[6*n+1];
    double *alpha_temp =  &recources[6*n+2];


    for (int k =0; k <= 100; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        cublasDdot(handle,n,dOld,1,rOld,1,alpha_temp);

        sp_matrix_times_V_ptrl(A,dOld,local_temp);
        cublasDdot(handle,n,local_temp,1,dOld,1,alpha_);

        //alpha_ = alpha_temp/alpha_;
        updateone<<<1,1>>>(alpha_temp,alpha_,alpha_);
        //         printf("alpha = %.6f\n",alpha_);exit(0);

        // xNew = xOld  + alpha.*dOld;
        updateXnew_<<<block,128>>>(xOld,dOld,xNew,alpha_,n);
        //rNew = b-A*xNew;
        //matrix_timesV(handle,A,xNew,n,n,rNew);
        sp_matrix_times_V_ptrl(A,xNew,rNew);

        update_rew<<<block,128>>>(rNew,b,n);
        cublasDnrm2(handle,n,rNew,1,&resNorm);

        //std::cout << resNorm << std::endl;
  
        if (resNorm < tol * bNorm){

            cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

            return ;

        }



        cublasDdot(handle,n,rNew,1,rNew,1,alpha_temp);
        cublasDdot(handle,n,rOld,1,rOld,1,alpha_);
        
        //rOld = rNew;
        cudaMemcpy(rOld,rNew,size_x,cudaMemcpyDeviceToDevice);

        //xOld = xNew;
        cudaMemcpy(xOld,xNew,size_x,cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

}

template <typename T>
void show_res_T(T *s,int n){
    T *res = (T *)malloc(n*sizeof(T));

    cudaMemcpy(res,s,n*sizeof(T),cudaMemcpyDeviceToHost);

    for (int i = 0 ; i< n;++i){
        printf("%.8f\t",res[i]);
    }

    printf("\n");
    free(res);
}

int main(int argc, char **argv){

   // read sparse matrix
    std::vector<std::string>  fieldes_sparse = {"A"};
    int i  = 0;

    COOMatrix A = COOMatrix();
    
    std::string f = "./" + std::to_string(i) +".txt";
    A.read_fromfile(f);
    std::cout << fieldes_sparse[i] << " " << A.n_cols << " " << A.n_rows << std::endl;

    COOMatrix &AAA = A;
    int equation_size = AAA.n_cols;

    double * bbb;
    double * yy;
    double * buff;

    cudaMalloc((void**)&bbb,sizeof(double) * equation_size);
    cudaMalloc((void**)&yy,sizeof(double) * equation_size);
    
    cudaMalloc((void**)&buff,sizeof(double) * 8 * equation_size);

    std::vector<double> b(equation_size,0);
    std::vector<double> zero(equation_size,0);

    std::default_random_engine e;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for(int i=0; i<equation_size; ++i)
        b[i] = u(e) ;
	
    cudaMemcpy(bbb,b.data(),sizeof(double) * equation_size,cudaMemcpyHostToDevice);

    //COOMatrix   AAA_coo(AAA);
    CSRMatrix   AAA_cu( AAA.n_element,AAA.n_rows,AAA.n_cols);
    AAA_cu.create_matrix_from_coo(&AAA);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(bbb,b.data(),sizeof(double) * equation_size,cudaMemcpyHostToDevice);  
    cudaMemcpy(buff,zero.data(),sizeof(double) * equation_size,cudaMemcpyHostToDevice); 
    cudaMemcpy(yy,zero.data(),sizeof(double) * equation_size,cudaMemcpyHostToDevice);  
    AAA_cu.create_matrix_from_coo(&AAA);

    auto start = std::chrono::steady_clock::now();
    //std::cout << "iter" << std::endl;
    for (int iter = 1; iter < 1000; iter++){
      
      Conjugate_gradient_sp(handle,&AAA_cu,bbb,yy,1e-1/((double)iter),1e3,buff,equation_size);
      //std::cout << iter << std::endl;
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time interval : " << diff.count() << " s\n";

    printf("\nPrinting Result: \n");
    show_res_T(yy, equation_size);

    return  0;

}

