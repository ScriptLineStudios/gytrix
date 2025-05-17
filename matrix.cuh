#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <float.h>

typedef struct {
    size_t rows, cols;
    float *elements;
    bool gpu;
} Mat;

typedef struct {
    size_t cols;
    float *elements;
} Row;

typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_COPY
} OpType; 

__device__ __host__ Mat mat_alloc(size_t rows, size_t cols);
__device__ __host__ void mat_fill(Mat m, float x);
__device__ __host__ void mat_rand(Mat m, float low, float high);
__device__ __host__ Row mat_row(Mat m, size_t row);
__device__ __host__ void mat_copy(Mat dst, Mat src);
__device__ __host__ void mat_dot(Mat dst, Mat a, Mat b);
__device__ __host__ void mat_sum(Mat dst, Mat a);
__device__ __host__ void mat_sub(Mat dst, Mat a);
__device__ __host__ void mat_mul(Mat dst, float x);
__device__ __host__ void mat_mul_mat(Mat a, Mat b);
__host__ void mat_print(Mat m, const char *name, size_t padding);
__device__ __host__ Mat mat_transpose(Mat m);
__device__ __host__ void mat_transpose_dst(Mat dst, Mat m);
__device__ __host__ void mat_free(Mat m);
Mat mat_to_cpu(Mat m);
Mat mat_to_gpu(Mat m);

#define MAT_AT(m, i, j) (m).elements[(i)*(m).cols + (j)]
#define ROW_AT(row, col) (row).elements[col]
#define MAT_PRINT(m) mat_print(m, #m, 0)
#endif

#ifdef GPU_MATRIX_IMPLEMENTATION
uint64_t __nano(void) {
   struct timespec currenttime;
   clock_gettime(CLOCK_REALTIME, &currenttime);
   return UINT64_C(1000000000) * currenttime.tv_sec + currenttime.tv_nsec;
}                   

static __device__ __inline__ uint64_t __nano_gpu(){                                                                         
  uint64_t mclk;                                                                                                        
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));                                                              
  return mclk ;                                                                                                         
} 
                                                                                                                        
__device__ __host__ float rand_float(uint64_t *rand) {
    (*rand) = (*rand) * 0x5DEECE66DL + 0xBL;
    return  (*rand) / (float) RAND_MAX;
}

__device__ __host__ Row mat_row(Mat m, size_t row) {
    return (Row) {
        .cols=m.cols,
        .elements= &MAT_AT(m, row, 0)
    };
}

__device__ __host__ float mat_at(Mat m, int x, int y) {
    return MAT_AT(m, y, x);
}

__host__ void mat_print(Mat m, const char *name, size_t padding) {
    Mat m1 = mat_to_cpu(m);
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m1, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
    mat_free(m1);
}

#define MAX(a,b) (((a)>=(b))?(a):(b))

__device__ __host__ int floordiv(int a, int b) {
    int32_t q = a / b;
    int32_t r = a % b;
    return q - ((a ^ b) < 0 && !!r);
}

#include <stdint.h>
#include <inttypes.h>

__global__ void mat_rand_internal(Mat m, float low, float high, int offset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (i >= m.rows * m.cols) {
        return;
    }
    uint64_t seed = ((__nano_gpu()+(i * 298319289381)) * 2089183908011192 + 11);
    m.elements[i] = ((float)seed / (float)UINT64_MAX) * (high - low) + low; 
}   

__device__ __host__ void mat_rand(Mat m, float low, float high) {
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= m.rows * m.cols; i += size) {
        mat_rand_internal<<<blocks, threads>>>(m, low, high, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
}

__global__ void mat_dot_internal(Mat output, Mat m1, Mat m2, int offset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    int x = i % m2.cols;
    int y = (int)i / (int)m2.cols;
    if (i >= m1.rows * m2.cols) {
        return;
    }
    output.elements[i] = 0;
    Row row = mat_row(m1, y);
    for (int c = 0; c < m2.rows; c++) {
        output.elements[i] += ROW_AT(row, c) * MAT_AT(m2, c, x);
    }
}   

__device__ __host__ void mat_dot(Mat dst, Mat a, Mat b) {
    assert(a.cols == b .rows);
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= a.rows * b.cols; i += size) {
        mat_dot_internal<<<blocks, threads>>>(dst, a, b, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
}

__global__ void mat_mat_op_internal(Mat a, Mat b, int offset, OpType op) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    int x = i % a.cols;
    int y = (int)i / (int)a.cols;
    if (i >= a.rows * a.cols) {
        return;
    }
    switch (op) {
        case OP_MUL:
            MAT_AT(a, y, x) = MAT_AT(a, y, x) * MAT_AT(b, y, x);
            return;
        case OP_ADD:
            MAT_AT(a, y, x) = MAT_AT(a, y, x) + MAT_AT(b, y, x);
            return;
        case OP_SUB:
            MAT_AT(a, y, x) = MAT_AT(a, y, x) - MAT_AT(b, y, x);
            return;
        case OP_DIV:
            MAT_AT(a, y, x) = MAT_AT(a, y, x) / MAT_AT(b, y, x);
            return;
        case OP_COPY:
            MAT_AT(a, y, x) = MAT_AT(b, y, x);
            return;
        default:
            assert(false);
    }
}   

__device__ __host__ void mat_mat_op(Mat a, Mat b, OpType op) {
    assert(a.rows == b.rows && a.cols == b.cols);
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= a.rows * b.cols; i += size) {
        mat_mat_op_internal<<<blocks, threads>>>(a, b, i, op);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
}

__device__ __host__ void mat_mul_mat(Mat a, Mat b) {
    mat_mat_op(a, b, OP_MUL);
}

__device__ __host__ void mat_sub(Mat a, Mat b) {
    mat_mat_op(a, b, OP_SUB);
}

__device__ __host__ void mat_sum(Mat a, Mat b) {
    mat_mat_op(a, b, OP_ADD);
}

__device__ __host__ void mat_div(Mat a, Mat b) {
    mat_mat_op(a, b, OP_DIV);
}

__device__ __host__ void mat_copy(Mat dst, Mat src) {
    mat_mat_op(dst, src, OP_COPY);
}

__global__ void mat_mul_internal(Mat a, float v, int offset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    int x = i % a.cols;
    int y = (int)i / (int)a.cols;
    if (i >= a.rows * a.cols) {
        return;
    }
    MAT_AT(a, y, x) = MAT_AT(a, y, x) * v;
}

__device__ __host__ void mat_mul(Mat dst, float x) {
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= dst.rows * dst.cols; i += size) {
        mat_mul_internal<<<blocks, threads>>>(dst, x, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
}

__global__ void mat_transpose_internal(Mat dst, Mat m1, int offset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    int x = i % dst.cols;
    if (i >= dst.cols) {
        return;
    }
    Row row = mat_row(m1, x);
    for (int i = 0; i < dst.rows; i++) {
        MAT_AT(dst, i, x) = ROW_AT(row, i);
    }
}

__device__ __host__ Mat mat_transpose(Mat m) {
    Mat dst = mat_alloc(m.cols, m.rows);
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= dst.rows * dst.cols; i += size) {
        mat_transpose_internal<<<blocks, threads>>>(dst, m, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
    return dst;
}   

__device__ __host__ void mat_transpose_dst(Mat dst, Mat m) {
    int threads = 256;
    int blocks = 262144;
    int size = threads * blocks;
    for (int i = 0; i <= dst.rows * dst.cols; i += size) {
        mat_transpose_internal<<<blocks, threads>>>(dst, m, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            assert(false);
        }
    }
    cudaDeviceSynchronize();
}   

__device__ __host__ Mat mat_alloc(size_t rows, size_t cols) {
    float *ret_elements;

    cudaError_t err = cudaMalloc(&ret_elements, sizeof(float) * rows * cols);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        assert(false);
    }

    // cudaMalloc(&ret_elements, sizeof(float) * rows * cols);
    // if (ret_elements == NULL) {
    //     assert(false);
    // }
    Mat m = (Mat){.rows=rows, .cols=cols, .elements=ret_elements, .gpu=true};
    return m;
}

Mat mat_to_cpu(Mat m) {
    if (!m.gpu) {
        assert(false);
    }
    float *ret_elements = (float *)malloc(sizeof(float) * m.rows * m.cols);
    assert(ret_elements != NULL);
    cudaMemcpy(ret_elements, m.elements, sizeof(float) * m.rows * m.cols, cudaMemcpyDeviceToHost);
    return (Mat){.rows=m.rows, .cols=m.cols, .elements=ret_elements, .gpu=false};
}

Mat mat_to_gpu(Mat m) {
    if (m.gpu) {
        assert(false);
    }
    Mat r = mat_alloc(m.rows, m.cols);
    cudaMemcpy(r.elements, m.elements, sizeof(float) * m.rows * m.cols, cudaMemcpyHostToDevice);
    return r;
}

__device__ __host__ void mat_free(Mat m) {
    if (m.gpu) {
        cudaFree(m.elements);
    }
    else {
        free(m.elements);
    }
}

void mat_save(Mat m, FILE *file) {
    Mat cm = mat_to_cpu(m);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            fwrite((const void *)&MAT_AT(cm, i, j), sizeof(float), 1, file);
        }
    }
    mat_free(cm);
}

Mat mat_load(size_t row, size_t col, FILE *file) {
    Mat tmp = mat_alloc(row, col);
    Mat m = mat_to_cpu(tmp);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            float value;
            size_t ret = fread((void *)&value, sizeof(float), 1, file);
            (void)ret;
            MAT_AT(m, i, j) = value;
        }
    }
    Mat ret = mat_to_gpu(m);
    mat_free(m);
    mat_free(tmp);
    return ret;
}
#endif