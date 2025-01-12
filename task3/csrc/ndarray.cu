#include<ndarray.h>

int NDArray::shape2size(const shape_t shape) {
    int res = 1;
    for (auto x: shape) {
        res *= x;
    }
    return res;
}

NDArray::NDArray(const shape_t shape, Device device): shape(shape), device(device) {
    this->size = NDArray::shape2size(shape);
    if (device == CPU) {
        this->p = new scalar_t[this->size];
        memset(this->p, 0, this->size * sizeof(scalar_t));
    } else {
        cudaMalloc(&this->p, this->size * sizeof(scalar_t));
        cudaMemset(this->p, 0, this->size * sizeof(scalar_t));
    }
}

NDArray::NDArray(const shape_t shape, scalar_t val, Device device): shape(shape), device(device) {
    this->size = NDArray::shape2size(shape);
    if (device == CPU) {
        this->p = new scalar_t[this->size];
        for (int i = 0; i < this->size; i++) {
            this->p[i] = val;
        }
    } else {
        cudaMalloc(&this->p, this->size * sizeof(scalar_t));    //!注意malloc只支持int
        thrust::device_ptr<scalar_t> p(this->p);
        thrust::fill(p, p + this->size, val);
    }
}

NDArray::NDArray(const NDArray& src): shape(src.shape), size(src.size), device(src.device) {
    if (this->device == CPU) {
        this->p = new scalar_t[this->size];
        memcpy(this->p, src.p, this->size * sizeof(scalar_t));
    } else {
        cudaMalloc(&this->p, this->size * sizeof(scalar_t));
        cudaMemcpy(this->p, src.p, this->size * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
    }
}

NDArray& NDArray::from_array(const array_t& src, Device device){
    NDArray* dst = new NDArray(shape_t(src.shape(), src.shape() + src.ndim()), device);
    if (device == CPU) {
        memcpy(dst->p, src.data(), dst->size * sizeof(scalar_t));
    } else {
        cudaMemcpy(dst->p, src.data(), dst->size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    }
    return *dst;
}

list_t& NDArray::tolist() {
    list_t* res = new list_t(this->size);
    if (this->device == CPU) {
        memcpy(res->data(), this->p, this->size * sizeof(scalar_t));
    } else {
        cudaMemcpy(res->data(), this->p, this->size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    }
    return *res;
}

/* Generate a random NDArray ~ U[0,1] of specific shape on the device. */
NDArray& NDArray::rand(const shape_t shape, scalar_t l, scalar_t h, Device device) {
    NDArray* dst = new NDArray(shape, GPU);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateUniform(gen, dst->p, dst->size);
    thrust::device_ptr<scalar_t> p(dst->p);
    thrust::transform(p, p + dst->size, p, [=] __device__ (scalar_t x) { return x * (h - l) + l; });
    curandDestroyGenerator(gen);
    return dst->to(device);
}

/* Generate a random NDArray ~ N(mean, std) of specific shape on the device. */
NDArray& NDArray::randn(const shape_t shape, scalar_t mean, scalar_t std, Device device) {
    NDArray* dst = new NDArray(shape, GPU);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    if (dst->size % 2 == 0) {   //!curandGenerateNormal only support even number
        curandGenerateNormal(gen, dst->p, dst->size, mean, std);
    } else {
        curandGenerateNormal(gen, dst->p, dst->size-1, mean, std);
        NDArray* tmp = new NDArray({2}, GPU);
        curandGenerateNormal(gen, tmp->p, 2, mean, std);
        cudaMemcpy(dst->p + dst->size - 1, tmp->p, sizeof(scalar_t), cudaMemcpyDeviceToDevice);
        delete tmp;
    }
    return dst->to(device);
}

NDArray& NDArray::operator = (const NDArray& src) {
    if (this == &src) {
        return *this;
    }
    // deep copy
    if (this->device == CPU) {
        delete[] this->p;
    } else {
        cudaFree(this->p);
    }
    this->shape = src.shape;
    this->size = src.size;
    this->device = src.device;
    if (this->device == CPU) {
        this->p = new scalar_t[this->size];
        memcpy(this->p, src.p, this->size * sizeof(scalar_t));
    } else {
        cudaMalloc(&this->p, this->size * sizeof(scalar_t));
        cudaMemcpy(this->p, src.p, this->size * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

NDArray::~NDArray() {
    if (device == CPU) {
        delete[] this->p;
    } else {
        cudaFree(this->p);
    }
}

NDArray& NDArray::cpu() {
    if (this->device == CPU) {
        return *this;
    } else {
        NDArray* tensor = new NDArray(this->shape, CPU);
        cudaMemcpy(tensor->p, this->p, this->size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        return *tensor;
    }
}

NDArray& NDArray::gpu() {
    if (this->device == GPU) {
        return *this;
    } else {
        NDArray* tensor = new NDArray(this->shape, GPU);
        cudaMemcpy(tensor->p, this->p, this->size * sizeof(scalar_t), cudaMemcpyHostToDevice);
        return *tensor;
    }
}

NDArray& NDArray::to(Device device) {
    if (this->device == device) {
        return *this;
    }
    if (device == CPU) {
        return this->cpu();
    } else {
        return this->gpu();
    }
}

NDArray& NDArray::operator + (const NDArray& B) {
    assert(this->shape == B.shape);
    NDArray* C = new NDArray(this->shape, this->device);
    thrust::device_ptr<scalar_t> A_ptr(this->p), B_ptr(B.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + this->size, B_ptr, C_ptr, thrust::plus<scalar_t>());
    return *C;
}

NDArray& NDArray::operator - (const NDArray& B) {
    assert(this->shape == B.shape);
    NDArray* C = new NDArray(this->shape, this->device);
    thrust::device_ptr<scalar_t> A_ptr(this->p), B_ptr(B.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + this->size, B_ptr, C_ptr, thrust::minus<scalar_t>());
    return *C;
}

NDArray& NDArray::operator / (const NDArray& B) {
    assert(this->shape == B.shape);
    NDArray* C = new NDArray(this->shape, this->device);
    thrust::device_ptr<scalar_t> A_ptr(this->p), B_ptr(B.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + this->size, B_ptr, C_ptr, thrust::divides<scalar_t>());
    return *C;
}

NDArray& NDArray::operator * (const NDArray& B) {
    //https://blog.csdn.net/weixin_52027058/article/details/136652593
    assert(this->shape.size() == 2 && B.shape.size() == 2);
    assert(this->shape[1] == B.shape[0]);
    int m = this->shape[0], 
        k = this->shape[1], 
        n = B.shape[1];
    int lda = n, ldb = k, ldc = n;
    NDArray* C = new NDArray({m, n}, this->device);
    const scalar_t alpha = 1, beta = 0;
    const scalar_t* p_alpha = &alpha;
    const scalar_t* p_beta = &beta;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, p_alpha,
        B.p, lda, this->p, ldb, p_beta, C->p, ldc);
    cublasDestroy(handle);
    return *C;
}

NDArray& NDArray::reshape(const shape_t new_shape) {
    assert(this->size == NDArray::shape2size(new_shape));
    NDArray* dst = new NDArray(*this);
    dst->shape = new_shape;
    return *dst;
}

__global__ void T2d_kernel(int m, int n, const scalar_t* src, scalar_t* dst) {
    int i = blockIdx.x, 
        j = threadIdx.x;
    dst[j * m + i] = src[i * n + j];
}

NDArray& NDArray::T2d() {
    assert(this->shape.size() == 2);
    int m = this->shape[0], n = this->shape[1];
    NDArray* dst = new NDArray({n, m}, this->device);
    if (this->device == CPU) {
        for (int i = 0; i < this->shape[0]; i++) {
            for (int j = 0; j < this->shape[1]; j++) {
                dst->p[j * this->shape[0] + i] = this->p[i * this->shape[1] + j];
            }
        }
    } else { 
        T2d_kernel<<<m, n>>>(m, n, this->p, dst->p);
        cudaDeviceSynchronize();
    }
    return *dst;
}

__global__ void swap_kernel(
    const int dim, const int size, 
    const int* old_shape, const int* new_shape, 
    const int axis1, const int axis2, 
    const scalar_t* src, scalar_t* dst) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int new_idx = 0, base = 1, old_idx = idx;
        int pos[MAX_DIMS];
        for (int i = dim-1; i >= 0; i--) {
            pos[i] = idx % old_shape[i];
            idx /= old_shape[i];
        }
        
        int tmp = pos[axis1];
        pos[axis1] = pos[axis2];
        pos[axis2] = tmp;

        for (int i=dim-1; i>=0; i--) {
            new_idx += pos[i] * base;
            base *= new_shape[i];
        }
        dst[new_idx] = src[old_idx];
    }
}

NDArray& NDArray::swap(const int axis1, const int axis2) {
    int dim = this->shape.size(), size = this->size;
    NDArray* dst = new NDArray(*this);

    int tmp = dst->shape[axis1];
    dst->shape[axis1] = dst->shape[axis2];
    dst->shape[axis2] = tmp;

    int* old_shape, *new_shape;
    cudaMalloc(&old_shape, dim * sizeof(int));
    cudaMemcpy(old_shape, this->shape.data(), dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&new_shape, dim * sizeof(int));
    cudaMemcpy(new_shape, dst->shape.data(), dim * sizeof(int), cudaMemcpyHostToDevice);
    swap_kernel<<<(this->size+255)/256, 256>>>(
        dim, size,
        old_shape, new_shape, 
        axis1, axis2, 
        this->p, dst->p);
    return *dst;
}

NDArray& mul(const NDArray& A, const NDArray& B) {
    assert(A.shape == B.shape);
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), B_ptr(B.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, B_ptr, C_ptr, thrust::multiplies<scalar_t>());
    return *C;
}

NDArray& mul_scalar(const NDArray& A, const scalar_t B) {
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, C_ptr, [=] __device__ (scalar_t x) { return x * B; });
    return *C;
}

NDArray& add_scalar(const NDArray& A, const scalar_t B) {
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, C_ptr, [=] __device__ (scalar_t x) { return x + B; });
    return *C;
}

NDArray& sub_scalar(const NDArray& A, const scalar_t B) {
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, C_ptr, [=] __device__ (scalar_t x) { return x - B; });
    return *C;
}

NDArray& div_scalar(const NDArray& A, const scalar_t B) {
    assert (B != 0);
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, C_ptr, [=] __device__ (scalar_t x) { return x / B; });
    return *C;
}

NDArray& pow_scalar(const NDArray& A, const scalar_t B) {
    NDArray* C = new NDArray(A.shape, A.device);
    thrust::device_ptr<scalar_t> A_ptr(A.p), C_ptr(C->p);
    thrust::transform(A_ptr, A_ptr + A.size, C_ptr, [=] __device__ (scalar_t x) { return pow(x, B); });
    return *C;
}
