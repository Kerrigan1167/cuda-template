#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define MIN(a,b) ({ \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#define MAX(a,b) ({ \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define DIV(a,b) \
    ({((a) % (b) == 0) ? ((a) / (b)) : ((a) / (b) + 1); })

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                \
    cudaError err = call;                                                \
    if (cudaSuccess != err) {                                            \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                              \
    } } while(0)

#define HOST_CHECK_POINTER(p) ({                                         \
    __typeof__ (p) __HOST_TEMP_POINTER = (p);                            \
    (__HOST_TEMP_POINTER == NULL) ? ({                                   \
        fprintf(stderr, "malloc error in file '%s' in line %i.\n",       \
                __FILE__, __LINE__);                                     \
        exit(EXIT_FAILURE);                                              \
        __HOST_TEMP_POINTER;                                             \
    }) :                                                                 \
    __HOST_TEMP_POINTER; })


__global__ void kernelTemplate(int *array, int arrayLength) {
    __shared__ int cache[1024];

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (offset < arrayLength) {
        cache[threadIdx.x] = array[offset];
    }
    __syncthreads();

    cache[threadIdx.x] ++;

    if (offset < arrayLength) {
        array[offset] = cache[threadIdx.x];
    }
}

int main(int argc, char **argv) {
    int array_length = 10000;
    
    int *array = HOST_CHECK_POINTER((int *)malloc(array_length * sizeof(int)));
    memset(array, 0, sizeof(array));

    int *dev_array;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&dev_array, array_length * sizeof(int)));
    
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dev_array, array, array_length * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(DIV(array_length, 1024));
    dim3 threadsPerBlock(1024);

    cudaEvent_t timer_start, timer_end;
    CUDA_SAFE_CALL_NO_SYNC(cudaEventCreate(&timer_start));
    CUDA_SAFE_CALL_NO_SYNC(cudaEventCreate(&timer_end));
    CUDA_SAFE_CALL_NO_SYNC(cudaEventRecord(timer_start, 0));
    
    kernelTemplate<<<blocksPerGrid, threadsPerBlock>>>(dev_array, array_length);
    CUDA_SAFE_CALL_NO_SYNC(cudaPeekAtLastError());  
    CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

    CUDA_SAFE_CALL_NO_SYNC(cudaEventRecord(timer_end, 0));
    CUDA_SAFE_CALL_NO_SYNC(cudaEventSynchronize(timer_end));

    float timer_elapsed;
    CUDA_SAFE_CALL_NO_SYNC(cudaEventElapsedTime(&timer_elapsed, timer_start, timer_end));
    printf ("Time: %3.1f ms\n", timer_elapsed);
    
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(array, dev_array, array_length * sizeof(int), cudaMemcpyDeviceToHost));
    
    int flag = 0;
    for (int i = 0; i < array_length; i++) {
        if (array[i] != 1) {
            printf ("kernel failed\n");
            flag ++;
            break;
        }
    }

    if (flag == 0)
        printf ("kernel sucessed\n");
    
    free(array);
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(dev_array));
    
    return 0;
}
