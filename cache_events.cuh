#include <cuda/atomic>
#include <iostream>
#include <chrono>
#include <fstream>
#include <thread>
#include <numa.h>
#include <unistd.h>
#include <string.h>
#include <vector>


#define BUFFER_SIZE 16384
#define NUM_ITERATIONS 20000

#ifndef DATA_SIZE
#define DATA_SIZE uint64_t
#endif

#ifndef CUDA_THREAD_SCOPE
#define CUDA_THREAD_SCOPE cuda::thread_scope_system
#endif

typedef enum {
    CE_SYS_MALLOC,
    CE_CUDA_MALLOC,
    CE_NUMA_HOST,
    CE_NUMA_DEVICE,
    CE_DRAM,
    CE_UM
} AllocatorType;

typedef enum {
    CE_GPU,
    CE_CPU
} ReaderWriterType;

long long duration_in_nanoseconds(unsigned long duration, int gpu_clock_rate) {
    return (long long) duration * 1000000000 / gpu_clock_rate;
}

// __device__ void do_stress(uint* scratchpad, uint* scratch_locations, uint iterations, uint pattern) {
//     for (uint i = 0; i < iterations; i++) {
//       if (pattern == 0) {
//         scratchpad[scratch_locations[blockIdx.x]] = i;
//         scratchpad[scratch_locations[blockIdx.x]] = i + 1;
//       }
//       else if (pattern == 1) {
//         scratchpad[scratch_locations[blockIdx.x]] = i;
//         uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
//         if (tmp1 > 100) {
//           break;
//         }
//       }
//       else if (pattern == 2) {
//         uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
//         if (tmp1 > 100) {
//           break;
//         }
//         scratchpad[scratch_locations[blockIdx.x]] = i;
//       }
//       else if (pattern == 3) {
//         uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
//         if (tmp1 > 100) {
//           break;
//         }
//         uint tmp2 = scratchpad[scratch_locations[blockIdx.x]];
//         if (tmp2 > 100) {
//           break;
//         }
//       }
//     }
//   }
  

__device__ static inline void __attribute__((optimize("O0"))) cudaSleep(clock_t sleep_cycles)
{
    volatile long long start = clock64();
    volatile long long cycles_elapsed;

    volatile int a, b;

    a = 17;
    b = 23;

    for (cycles_elapsed = clock64() - start; cycles_elapsed < sleep_cycles; cycles_elapsed = clock64() - start) {

        a = a * b;
        b = 3 * a + b * b + 4;
        a = a + b;
        b = b - a;
        
        // printf("cycles_elapsed = %ld sleep_cycles = %ld \n", cycles_elapsed, sleep_cycles);
    } 

    cycles_elapsed += a + b;
}

static void __attribute__((optimize("O0"))) buffer_reader_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> * buffer) {

    int local_buffer = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        local_buffer = local_buffer + buffer[i].load(cuda::memory_order_relaxed);
    }
    printf("[SUM] %d\n", local_buffer);
}

// static void __attribute__((optimize("O0"))) buffer_reader_single_iter(DATA_SIZE * buffer) {

//     DATA_SIZE local_buffer = 0;

//     for (DATA_SIZE i = 0; i < BUFFER_SIZE; i++) {
//         local_buffer = local_buffer + buffer[i];
//     }
//     printf("%lu\n", local_buffer);
// }

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, DATA_SIZE *results, clock_t *duration) {

    DATA_SIZE local_results = 0;

    clock_t begin = clock64();
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            local_results = local_results + buffer[j].load(cuda::memory_order_relaxed);
        }
    }
    clock_t end = clock64();
    *duration = end - begin;
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, DATA_SIZE *results, clock_t *duration) {

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        results[i] = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_t begin = clock64();
        for (int j = 0; j < BUFFER_SIZE; j++) {
            results[i] = results[i] + buffer[j].load(cuda::memory_order_relaxed);
        }
        clock_t end = clock64();
        duration[i] = end - begin;
        printf("[GPU-R] Iter %d Sum %lu Time %lu\n", i, results[i], duration[i]);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_diverge(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, DATA_SIZE *results, clock_t *duration) {

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        results[i] = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_t begin = clock64();
        for (int j = 0; j < BUFFER_SIZE; j++) {
            results[i] = results[i] + buffer[j].load(cuda::memory_order_relaxed);
        }
        clock_t end = clock64();
        duration[i] = end - begin;
        printf("[GPU-R] Iter %d Sum %lu Time %lu\n", i, results[i], duration[i]);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, int chunkSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);
    
    for (int i = start; i < end; i++) {
        buffer[i].store(1, cuda::memory_order_relaxed);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, int chunkSize, clock_t *sleep_duration) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);

    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = start; i < end; ++i) {
            buffer[i].store(j+1, cuda::memory_order_relaxed);
        }
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_diverge(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, clock_t *sleep_duration) {
    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int start = (threadId - 1) * chunkSize;
    // int end = min(start + chunkSize, BUFFER_SIZE);

    cudaSleep(*sleep_duration);

    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buffer[i].store(j+1, cuda::memory_order_relaxed);
        }
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, clock_t *sleep_duration, DATA_SIZE *results, clock_t *duration) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        gpu_buffer_reader_diverge(buffer, results, duration);
    } else {
        gpu_buffer_writer_diverge(buffer, sleep_duration);
    }

}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_thread(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, int chunkSize, clock_t *sleep_duration) {
    
    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buffer[i].store(j+1, cuda::memory_order_relaxed);
        }
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

static void __attribute__((optimize("O0"))) cpu_buffer_writer_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].store(1, cuda::memory_order_relaxed);
    }
}

static void __attribute__((optimize("O0"))) cpu_buffer_writer(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, struct timespec * sleep_duration) {
    
    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[CPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i].store(j+1, cuda::memory_order_relaxed);
        }
        printf("[CPU-W] Stop Iter %d\n", j);
        nanosleep(sleep_duration, NULL);
    }
}

static void __attribute__((optimize("O0"))) cpu_buffer_reader_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {
    
    int local_buffer = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        local_buffer = local_buffer + buffer[i].load(cuda::memory_order_relaxed);
    }
}

static void __attribute__((optimize("O0"))) cpu_buffer_reader(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, DATA_SIZE * result, std::chrono::duration<uint64_t, std::nano> *duration) {

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        duration[i] = std::chrono::nanoseconds(0);
    }
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result[i] = result[i] + buffer[j].load();
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        duration[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("[CPU-R] Iter %d Sum %lu Time %lu\n", i, result[i], duration[i].count());
    }
}

static void __attribute__((optimize("O0"))) buffer_reader(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {

    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::high_resolution_clock::time_point end;

    unsigned long int duration = 0;

    static volatile int local_buffer[NUM_ITERATIONS];

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        local_buffer[i] = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {

        begin = std::chrono::high_resolution_clock::now();
        
        // DATA_SIZE x;

        for (int j = 0; j < BUFFER_SIZE; j++) {
            local_buffer[i] = local_buffer[i] + buffer[j].load();

            // std::cout << "buffer[" << j << "]: " << buffer[j].load(cuda::memory_order_relaxed) << std::endl;
        }

        // local_buffer[i] = x;

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        printf("[CPU ITER-%d] %lu %d\n", i, duration , local_buffer[i]);
    }
}

// static void __attribute__((optimize("O0"))) buffer_reader(DATA_SIZE *buffer) {

//     std::chrono::high_resolution_clock::time_point begin;
//     std::chrono::high_resolution_clock::time_point end;

//     DATA_SIZE duration = 0;

//     static volatile DATA_SIZE local_buffer[NUM_ITERATIONS];

//     for (DATA_SIZE i = 0; i < NUM_ITERATIONS; i++) {
//         local_buffer[i] = 0;
//     }

//     for (DATA_SIZE i = 0; i < NUM_ITERATIONS; i++) {

//         begin = std::chrono::high_resolution_clock::now();
        
//         // DATA_SIZE x;

//         for (DATA_SIZE j = 0; j < BUFFER_SIZE; j++) {
//             local_buffer[i] = local_buffer[i] + buffer[j];//.load();

//             // std::cout << "buffer[" << j << "]: " << buffer[j].load(cuda::memory_order_relaxed) << std::endl;
//         }

//         // local_buffer[i] = x;

//         end = std::chrono::high_resolution_clock::now();
//         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

//         printf("%f %lu\n", (float) duration / (float) BUFFER_SIZE, local_buffer[i]);
//     }
// }

__global__ void gpuTrigger(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, DATA_SIZE num, int chunkSize) {
    // for (int )

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);

    // DATA_SIZE stride = gridDim.x * blockDim.x;

    // for (int j = 0; j > -1; j++) {
    for (int i = start; i < end; ++i) {
        buffer[i].store(num, cuda::memory_order_relaxed);
    }
    // }
}

// __global__ void gpuTrigger(DATA_SIZE *buffer, DATA_SIZE num, int chunkSize) {
//     // for (int )

//     int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//     int start = threadId * chunkSize;
//     int end = min(start + chunkSize, BUFFER_SIZE);

//     // DATA_SIZE stride = gridDim.x * blockDim.x;

//     for (DATA_SIZE i = start; i < end; ++i) {
//         // buffer[i].store(num, cuda::memory_order_relaxed);
//         buffer[i] = num;
//     }
// }

int get_gpu_properties() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // std::cout << "Device name: " << prop.name << std::endl;
    // std::cout << "Total Global Memory: " << prop.totalGlobalMem << std::endl;
    // std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
    // std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    // std::cout << "Warp Size: " << prop.warpSize << std::endl;
    // std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Max Threads Dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    // std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    // std::cout << "Clock Rate: " << prop.clockRate << std::endl;
    // std::cout << "Total Constant Memory: " << prop.totalConstMem << std::endl;
    // std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    // std::cout << "Kernel Execution Timeout Enabled: " << prop.kernelExecTimeoutEnabled << std::endl;
    // std::cout << "Integrated: " << prop.integrated << std::endl;
    // std::cout << "Can Map Host Memory: " << prop.canMapHostMemory << std::endl;
    // std::cout << "Compute Mode: " << prop.computeMode << std::endl;
    // std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
    // std::cout << "ECC Enabled: " << prop.ECCEnabled << std::endl;
    // std::cout << "PCI Bus ID: " << prop.pciBusID << std::endl;
    // std::cout << "PCI Device ID: " << prop.pciDeviceID << std::endl;
    // std::cout << "TCC Driver: " << prop.tccDriver << std::endl;
    // std::cout << "Memory Clock Rate: " << prop.memoryClockRate << std::endl;
    // std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
    // std::cout << "L2 Cache Size: " << prop.l2CacheSize << std::endl;
    // std::cout << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "Stream Priorities Supported: " << prop.streamPrioritiesSupported << std::endl;
    // std::cout << "Global L1 Cache Supported: " << prop.globalL1CacheSupported << std::endl;
    // std::cout << "Local L1 Cache Supported: " << prop.localL1CacheSupported << std::endl;
    // std::cout << "Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    // std::cout << "Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    // std::cout << "Managed Memory: " << prop.managedMemory << std::endl;
    // std::cout << "Is Multi-GPU Board: " << prop.isMultiGpuBoard << std::endl;
    // std::cout << "Multi-GPU Board Group ID: " << prop.multiGpuBoardGroupID << std::endl;

    
    return prop.clockRate;
}
