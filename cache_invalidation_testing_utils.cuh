#ifndef CACHE_INVALIDATION_TESTING_UTILS_CUH
#define CACHE_INVALIDATION_TESTING_UTILS_CUH

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

#include "cache_invalidation_testing_looping.cuh"
#include "cache_invalidation_testing_propagation_radius.cuh"
#include "cache_invalidation_testing_propagation_hierarchy.cuh"

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

#endif