#include <cuda/atomic>
#include <iostream>
#include <chrono>
#include <fstream>
#include <thread>
#include <numa.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <sched.h>

#define BUFFER_SIZE 512
#define NUM_ITERATIONS 10000

#define GPU_NUM_BLOCKS 4
#define GPU_NUM_THREADS 64

#define CPU_NUM_THREADS 32

// #ifndef DATA_SIZE
// #define DATA_SIZE uint64_t
// #endif

#ifdef DATA_SIZE_64
typedef uint64_t DATA_SIZE;
#elif defined(DATA_SIZE_32)
typedef uint32_t DATA_SIZE;
#elif defined(DATA_SIZE_16)
typedef uint16_t DATA_SIZE;
#elif defined(DATA_SIZE_8)
typedef uint8_t DATA_SIZE;
#endif

// #ifndef CUDA_THREAD_SCOPE
// #define CUDA_THREAD_SCOPE cuda::thread_scope_system
// #endif

#ifdef CUDA_THREAD_SCOPE_THREAD
#define CUDA_THREAD_SCOPE cuda::thread_scope_thread
#elif defined(CUDA_THREAD_SCOPE_BLOCK)
#define CUDA_THREAD_SCOPE cuda::thread_scope_block
#elif defined(CUDA_THREAD_SCOPE_DEVICE)
#define CUDA_THREAD_SCOPE cuda::thread_scope_device
#elif defined(CUDA_THREAD_SCOPE_SYSTEM)
#define CUDA_THREAD_SCOPE cuda::thread_scope_system
#endif

#ifdef SIGNAL_THREAD_SCOPE_THREAD
#define SIGNAL_THREAD_SCOPE cuda::thread_scope_thread
#elif defined(SIGNAL_THREAD_SCOPE_BLOCK)
#define SIGNAL_THREAD_SCOPE cuda::thread_scope_block
#elif defined(SIGNAL_THREAD_SCOPE_DEVICE)
#define SIGNAL_THREAD_SCOPE cuda::thread_scope_device
#elif defined(SIGNAL_THREAD_SCOPE_SYSTEM)
#define SIGNAL_THREAD_SCOPE cuda::thread_scope_system
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

typedef enum {
    CE_NO_WRITER,
    CE_WRITER,
    CE_HET_WRITER
} WriterType;

typedef struct bufferElement {
    // DATA_SIZE data;
    cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> data;
    char padding[4096 - sizeof(DATA_SIZE)];
} bufferElement;

typedef struct bufferElement_na {
    uint32_t data;
    char padding[4096 - sizeof(uint32_t)];
} bufferElement_na;

/**
 * @brief The GPU does a busy wait, with actual work, to get 
 * close to that timespan. Don't trust the numbers though. 
 * 
 * @param sleep_cycles 
 * @author Soham Bagchi
 */
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


static void __attribute__((optimize("O0"))) buffer_reader_single_iter(bufferElement * buffer) {

    int local_buffer = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        local_buffer = local_buffer + buffer[i].data.load(cuda::memory_order_relaxed);
    }
    printf("[SUM] %d\n", local_buffer);
}


__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_single_iter(bufferElement *buffer, uint32_t *results, clock_t *duration) {

    // uint32_t local_results = 0;

    clock_t begin = clock64();
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            *results = *results + buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }
    clock_t end = clock64();
    *duration = end - begin;
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader(bufferElement *buffer, uint32_t *results, uint32_t *duration) {

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        results[i] = 0;
    }

    int k_ = 256;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_t begin = clock64();
        for (int k = 0; k < k_; k++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
                results[i] = results[i] + buffer[j].data.load(cuda::memory_order_relaxed);
            }
        }
        clock_t end = clock64();
        duration[i] = end - begin;
        printf("[GPU-R] Iter %d Sum %u Time %u | %u\n", i, results[i], duration[i], duration[i]/BUFFER_SIZE/k_/2);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_diverge(bufferElement *buffer, uint32_t *results, uint32_t *duration) {

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        results[i] = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_t begin = clock64();
        // for (int k = 0; k < NUM_ITERATIONS / 100; i++) {
            for (int j = 0; j < BUFFER_SIZE; j++) {
                results[i] = results[i] + buffer[j].data.load(cuda::memory_order_relaxed);
            }
        // }
        clock_t end = clock64();
        duration[i] = end - begin;
        printf("[GPU-R] Iter %d Sum %u Time %u\n", i, results[i], duration[i]);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_diverge_constant(bufferElement *buffer, uint32_t * result) {

    uint dummy_result = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        dummy_result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    uint results = 0;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // clock_t begin = clock64();
        // for (int k = 0; k < NUM_ITERATIONS / 100; i++) {
        results = 0;
        for (int j = 0; j < BUFFER_SIZE; j++) {
            results += buffer[j].data.load(cuda::memory_order_relaxed);
        }
        // *result = ((2 * *result) + *result) * *result;
        // }
        // clock_t end = clock64();
        // duration[i] = end - begin;
        // printf("[GPU-R] Iter %d Sum %u Time %u\n", i, results[i], duration[i]);
    }
    *result = results;
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_iter(bufferElement *buffer, int chunkSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);
    
    for (int i = start; i < end; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_iter_single_thread(bufferElement *buffer, int chunkSize) {
    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int start = threadId * chunkSize;
    // int end = min(start + chunkSize, BUFFER_SIZE);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer(bufferElement *buffer, int chunkSize, clock_t *sleep_duration) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);

    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = start; i < end; ++i) {
            buffer[i].data.store(j+1, cuda::memory_order_relaxed);
        }
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_diverge(bufferElement *buffer, clock_t *sleep_duration) {
    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int start = (threadId - 1) * chunkSize;
    // int end = min(start + chunkSize, BUFFER_SIZE);

    cudaSleep(*sleep_duration);

    for (int j = 0; j < NUM_ITERATIONS / 100; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE - 1; ++i) {
            buffer[i].data.store(j+1, cuda::memory_order_relaxed);
        }
        buffer[BUFFER_SIZE - 1].data.store(j+1, cuda::memory_order_release);
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_diverge_constant(bufferElement *buffer) {
    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int start = (threadId - 1) * chunkSize;
    // int end = min(start + chunkSize, BUFFER_SIZE);

    // cudaSleep(*sleep_duration);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    cudaSleep(10000000);

    for (int j = 0; j < NUM_ITERATIONS; j++) {
        // printf("[GPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buffer[i].data.store(j+1, cuda::memory_order_relaxed);
        }
        // buffer[BUFFER_SIZE - 1].store(j+1, cuda::memory_order_release);
        // printf("[GPU-W] Stop Iter %d\n", j);
        // cudaSleep(*sleep_duration);
    }
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer(bufferElement *buffer, bufferElement *w_buffer, clock_t *sleep_duration, uint32_t *results, uint32_t *duration) {

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    if (blockId == 0 && threadId == 0) {
        gpu_buffer_writer_diverge(w_buffer, sleep_duration);
    } else if (blockId == 1 && threadId == 0) {
        gpu_buffer_reader_diverge(buffer, results, duration);
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_writer_worker(bufferElement *buffer) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(i, cuda::memory_order_relaxed);
        }
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_reader_worker(bufferElement *buffer, bufferElement_na *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint result = 0;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result += buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }

    results[tid].data = result;
}


__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer_constant(bufferElement *buffer, bufferElement *w_buffer, uint32_t * result, clock_t * t_reader, clock_t * t_writer) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    if (blockId == 0 && threadId == 0) {
        
        clock_t writer_start = clock64();
        #ifdef BUFFER_SAME
        gpu_buffer_writer_diverge_constant(buffer);
        #else
        gpu_buffer_writer_diverge_constant(w_buffer);
        #endif
        clock_t writer_end = clock64();

        *t_writer = writer_end - writer_start;
    } else if (blockId == GPU_NUM_BLOCKS - 1 && threadId == GPU_NUM_THREADS - 1) {

        clock_t reader_start = clock64();
        gpu_buffer_reader_diverge_constant(buffer, result);
        clock_t reader_end = clock64();

        *t_reader = reader_end - reader_start;
    } else {
        gpu_dummy_writer_worker(w_buffer);
    }
}

// static void __attribute__((optimize("O0"))) cpu_buffer_writer_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {
    static void __attribute__((optimize("O0"))) cpu_buffer_writer_single_iter(bufferElement *buffer) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i].data.store(1, cuda::memory_order_relaxed);
        }
    }
    
    // static void __attribute__((optimize("O0"))) cpu_buffer_writer(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, struct timespec * sleep_duration) {
    static void __attribute__((optimize("O0"))) cpu_buffer_writer(bufferElement *buffer, struct timespec * sleep_duration) {
        
        for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
            printf("[CPU-W] Start Iter %d\n", j);
            for (int i = 0; i < BUFFER_SIZE; i++) {
                buffer[i].data.store(j+1, cuda::memory_order_relaxed);
            }
            printf("[CPU-W] Stop Iter %d\n", j);
            nanosleep(sleep_duration, NULL);
        }
    }
    
    // static void __attribute__((optimize("O0"))) cpu_buffer_reader_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {
    static void __attribute__((optimize("O0"))) cpu_buffer_reader_single_iter(bufferElement *buffer) {
        
        int local_buffer = 0;
        
        for (int i = 0; i < BUFFER_SIZE; i++) {
            local_buffer = local_buffer + buffer[i].data.load(cuda::memory_order_relaxed);
        }
    }
    
    // static void __attribute__((optimize("O0"))) cpu_buffer_reader(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, uint32_t * result, std::chrono::duration<uint32_t, std::nano> *duration) {
    static void __attribute__((optimize("O0"))) cpu_buffer_reader(bufferElement *buffer, uint32_t * result, std::chrono::duration<uint32_t, std::nano> *duration) {
    
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            duration[i] = std::chrono::nanoseconds(0);
        }
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < BUFFER_SIZE; j++) {
                result[i] = result[i] + buffer[j].data.load();
            }
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            duration[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            printf("[CPU-R] Iter %d Sum %u Time %u\n", i, result[i], duration[i].count());
        }
    }
    
    static void __attribute__((optimize("O0"))) buffer_reader(bufferElement *buffer) {
    
        std::chrono::high_resolution_clock::time_point begin;
        std::chrono::high_resolution_clock::time_point end;
    
        unsigned long int duration = 0;
    
        static volatile uint32_t local_buffer[NUM_ITERATIONS];
    
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            local_buffer[i] = 0;
        }
    
        for (int i = 0; i < NUM_ITERATIONS; i++) {
    
            begin = std::chrono::high_resolution_clock::now();
            
            // DATA_SIZE x;
    
            for (int j = 0; j < BUFFER_SIZE; j++) {
                local_buffer[i] = local_buffer[i] + buffer[j].data.load();
    
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
    

__global__ void gpuTrigger(bufferElement *buffer, DATA_SIZE num, int chunkSize) {
    // for (int )

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * chunkSize;
    int end = min(start + chunkSize, BUFFER_SIZE);

    // DATA_SIZE stride = gridDim.x * blockDim.x;

    // for (int j = 0; j > -1; j++) {
    for (int i = start; i < end; ++i) {
        buffer[i].data.store(num, cuda::memory_order_relaxed);
    }
    // }
}

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_writer_worker_propagation(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal) {

    r_signal->fetch_add(1, cuda::memory_order_relaxed);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        if (i % (NUM_ITERATIONS / 4) == 0) {
            cudaSleep(10000000000);
        }
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(i, cuda::memory_order_relaxed);
        }
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint result = 0;

    r_signal->fetch_add(1, cuda::memory_order_relaxed);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result += buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }

    results[tid].data = result;
}



__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *w_signal) {

    while(r_signal->load(cuda::memory_order_acquire) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }
    
    // for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(10, cuda::memory_order_relaxed);
        }
    // }

    // Set Writer Signal
    w_signal->store(1, cuda::memory_order_release);
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation_acq(bufferElement *buffer, bufferElement_na *results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *w_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;

    
    // Set Reader Signal
    r_signal->fetch_add(1, cuda::memory_order_relaxed);
    
    result = 0;
    while(w_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_acquire);
    }

    results[tid].data = result;
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation_rlx(bufferElement *buffer, bufferElement_na *results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *w_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;

    
    // Set Reader Signal
    r_signal->fetch_add(1, cuda::memory_order_relaxed);
    
    result = 0;
    while(w_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer_propagation_radius(bufferElement *buffer, bufferElement *w_buffer, bufferElement_na * results, clock_t * t_reader, clock_t * t_writer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *w_signal) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    if (blockId == 0 && threadId == 0) {
        
        clock_t writer_start = clock64();

        #ifdef BUFFER_SAME
        gpu_buffer_writer_propagation(buffer, r_signal, w_signal);
        #else
        gpu_buffer_writer_propagation(w_buffer, r_signal, w_signal);
        #endif

        clock_t writer_end = clock64();

        *t_writer = writer_end - writer_start;
    } else if (threadId == GPU_NUM_THREADS - 1) {

        // clock_t reader_start = clock64();
        gpu_buffer_reader_propagation_rlx(buffer, results, r_signal, w_signal);
        // clock_t reader_end = clock64();

        // *t_reader = reader_end - reader_start;
    } else if (blockId >= GPU_NUM_BLOCKS / 2 && threadId == 0) {
        gpu_buffer_reader_propagation_acq(buffer, results, r_signal, w_signal);
    } else if (blockId < GPU_NUM_BLOCKS / 2 && blockId != 0 && threadId == 0) {
        gpu_buffer_reader_propagation_rlx(buffer, results, r_signal, w_signal);
    } else if (threadId % 2 == 0){
        gpu_dummy_writer_worker_propagation(w_buffer, r_signal);
    } else if (threadId % 2 != 0) {
        gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
    }
    
}





template <typename T>
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation_hierarchy_acq(bufferElement * buffer, bufferElement_na * results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> * r_signal, T *w_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> * fallback_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;

    // Set Reader Signal
    r_signal->fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while(w_signal->load(cuda::memory_order_relaxed) == 0 && fallback_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_acquire);
    }

    results[tid].data = result;
    
    printf("B[%d] T[%d] (%d:%d:%d) Result %d\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4, results[tid].data);
}

template <typename T>
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation_hierarchy_rlx(bufferElement *buffer, bufferElement_na *results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, T *w_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Set Reader Signal
    r_signal->fetch_add(1, cuda::memory_order_relaxed);
    
    result = 0;
    
    while(w_signal->load(cuda::memory_order_relaxed) == 0 && fallback_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    // cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_device);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    printf("B[%d] T[%d] (%d:%d:%d) Result %d\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4, results[tid].data);
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation_hierarchy(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> * r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> * w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> * w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> * w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> * w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> * fallback_signal) {

    printf("GPU Writer %d\n", blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->load(cuda::memory_order_acquire) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    
    // Set Writer Signals (Thread)
    w_t_signal->store(1, cuda::memory_order_relaxed);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Block)
    w_b_signal->store(1, cuda::memory_order_relaxed);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Device)
    w_d_signal->store(1, cuda::memory_order_relaxed);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (System)

    w_s_signal->store(1, cuda::memory_order_relaxed);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    cudaSleep(10000000000);
    
    fallback_signal->store(1, cuda::memory_order_release);
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation_hierarchy_cpu(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> *w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> *w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> *w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal
) {
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("GPU Het-Writer %d\n", blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->load(cuda::memory_order_acquire) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Thread)
    w_t_signal->store(1, cuda::memory_order_relaxed);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Block)
    w_b_signal->store(1, cuda::memory_order_relaxed);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Device)
    w_d_signal->store(1, cuda::memory_order_relaxed);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (System)
    w_s_signal->store(1, cuda::memory_order_relaxed);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    cudaSleep(10000000000);

    fallback_signal->store(1, cuda::memory_order_release);
}

__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer_propagation_hierarchy(bufferElement *buffer, bufferElement *w_buffer, bufferElement_na * results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> *w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> *w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> *w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal, WriterType *spawn_writer) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    
    if ((*spawn_writer == CE_WRITER || *spawn_writer == CE_HET_WRITER) && blockId == 0 && threadId == 0) {
        printf("GPU Home\n");
        // gpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        if (*spawn_writer == CE_WRITER) {
            gpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            gpu_buffer_writer_propagation_hierarchy_cpu(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else if (blockId == 0) {
        if (threadId % 8 == 0) {
            // TODO: Relaxed Thread
            // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (threadId % 8 == 1) {
            // TODO: Relaxed Block
            // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (threadId % 8 == 2) {
            // TODO: Relaxed Device
            // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (threadId % 8 == 3) {
            // TODO: Relaxed System
            // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_s_signal, fallback_signal);
        } else if (threadId % 8 == 4) {
            // TODO: Acquire Thread
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (threadId % 8 == 5) {
            // TODO: Acquire Block
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (threadId % 8 == 6) {
            // TODO: Acquire Device
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (threadId % 8 == 7) {
            // TODO: Acquire System
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_s_signal, fallback_signal);
        }
    } else {
        if (threadId == 0) {
            gpu_dummy_writer_worker_propagation(w_buffer, r_signal);
        } else if (threadId < 16) {
            if (threadId % 8 == 0) {
                // TODO: Relaxed Thread
                // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_t_signal, fallback_signal);
            } else if (threadId % 8 == 1) {
                // TODO: Relaxed Block
                // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_b_signal, fallback_signal);
            } else if (threadId % 8 == 2) {
                // TODO: Relaxed Device
                // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_d_signal, fallback_signal);
            } else if (threadId % 8 == 3) {
                // TODO: Relaxed System
                // gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_s_signal, fallback_signal);
            } else if (threadId % 8 == 4) {
                // TODO: Acquire Thread
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_t_signal, fallback_signal);
            } else if (threadId % 8 == 5) {
                // TODO: Acquire Block
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_b_signal, fallback_signal);
            } else if (threadId % 8 == 6) {
                // TODO: Acquire Device
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_d_signal, fallback_signal);
            } else if (threadId % 8 == 7) {
                // TODO: Acquire System
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
                // gpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_s_signal, fallback_signal);
            }
        } else {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        }
    }
} 

// __global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_thread(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer, int chunkSize, clock_t *sleep_duration) {
__global__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_single_thread(bufferElement *buffer, int chunkSize, clock_t *sleep_duration) {
    
    for (int j = 0; j < NUM_ITERATIONS / 1000; j++) {
        printf("[GPU-W] Start Iter %d\n", j);
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buffer[i].data.store(j+1, cuda::memory_order_relaxed);
        }
        printf("[GPU-W] Stop Iter %d\n", j);
        cudaSleep(*sleep_duration);
    }
}

static void __attribute__((optimize("O0"))) cpu_buffer_writer_propagation_hierarchy(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> *w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> *w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> *w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal) {
    
    printf("CPU Writer %d\n", sched_getcpu());

    // int core_id = sched_getcpu();
    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->load(cuda::memory_order_acquire) != CPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    
    // Set Writer Signals (Thread)
    w_t_signal->store(1, cuda::memory_order_relaxed);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Block)
    w_b_signal->store(1, cuda::memory_order_relaxed);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Device)
    w_d_signal->store(1, cuda::memory_order_relaxed);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (System)
    w_s_signal->store(1, cuda::memory_order_relaxed);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    sleep(5);

    fallback_signal->store(1, cuda::memory_order_release);

}

template <typename R, typename W, typename F>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_propagation_hierarchy_acq(bufferElement *buffer, bufferElement_na * results, R *r_signal, W *w_signal, F *fallback_signal) {
    
    int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id].data = result;

    r_signal->fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while (w_signal->load(cuda::memory_order_relaxed) == 0 && fallback_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_acquire);
    }

    results[core_id % CPU_NUM_THREADS].data = result;

    printf("C[%d:%d:%d] Result %d\n", core_id, core_id % 8, core_id % 4, results[core_id % CPU_NUM_THREADS].data);
}

template <typename R, typename W, typename F>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_propagation_hierarchy_rlx(bufferElement *buffer, bufferElement_na * results, R *r_signal, W *w_signal, F *fallback_signal) {
    
    int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id].data = result;

    r_signal->fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while (w_signal->load(cuda::memory_order_relaxed) == 0 && fallback_signal->load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id % CPU_NUM_THREADS].data = result;

    printf("C[%d:%d:%d] Result %d\n", core_id, core_id % 8, core_id % 4, results[core_id % CPU_NUM_THREADS].data);
}

static void __attribute__((optimize("O0"))) cpu_buffer_writer_propagation_hierarchy_gpu(bufferElement *buffer, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> *w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> *w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> *w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal) {

    printf("CPU Het-Writer %d %d\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 1);
    // int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->load(cuda::memory_order_acquire) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    w_t_signal->store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_b_signal->store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_d_signal->store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_s_signal->store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    sleep(5);

    fallback_signal->store(1, cuda::memory_order_release);
}

template <typename R>
static void __attribute__((optimize("O0"))) cpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, R *r_signal) {
    int core_id = sched_getcpu();
    
    uint result = 0;
    
    r_signal->fetch_add(1, cuda::memory_order_relaxed);
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result += buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
}

static void __attribute__((optimize("O0"))) cpu_buffer_reader_writer_propagation_hierarchy(bufferElement *buffer, bufferElement *w_buffer, bufferElement_na * results, cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *r_signal, cuda::atomic<uint32_t, cuda::thread_scope_thread> *w_t_signal, cuda::atomic<uint32_t, cuda::thread_scope_block> *w_b_signal, cuda::atomic<uint32_t, cuda::thread_scope_device> *w_d_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *w_s_signal, cuda::atomic<uint32_t, cuda::thread_scope_system> *fallback_signal, WriterType *spawn_writer) {

    
    int core_id = sched_getcpu();
    
    if ((*spawn_writer == CE_WRITER || *spawn_writer == CE_HET_WRITER) && core_id == 0) {
        printf("CPU Home\n");
        // cpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        if (*spawn_writer == WriterType::CE_WRITER) {
            cpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            cpu_buffer_writer_propagation_hierarchy_gpu(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else {
        if (core_id % 8 == 0) {
            cpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (core_id % 8 == 1) {
            cpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (core_id % 8 == 2) {
            cpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (core_id % 8 == 3) {
            cpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_s_signal, fallback_signal);
        } else if (core_id % 8 == 4) {
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // cpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (core_id % 8 == 5) {
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // cpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (core_id % 8 == 6) {
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // cpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (core_id % 8 == 7) {
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            // cpu_buffer_reader_propagation_hierarchy_acq(buffer, results, r_signal, w_s_signal, fallback_signal);
        }
    }
}