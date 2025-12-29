/**
 * @file cache_invalidation_testing.cuh
 * @brief CUDA Cache Invalidation Penalty Testing Framework
 * 
 * Tests cache coherence propagation delays across heterogeneous CPU-GPU systems.
 * Measures how memory ordering semantics (acquire/release/relaxed) and CUDA thread
 * scopes (thread/block/device/system) affect cache invalidation propagation.
 * 
 * MAIN EXECUTION PATTERNS:
 * 1. Single Writer Propagation Hierarchy - One writer, multiple readers test
 *    visibility across scope levels (thread → block → device → system)
 * 2. Multi-Writer Propagation Hierarchy - Four concurrent writers test
 *    concurrent visibility at different scope levels
 * 
 * TIMING INSTRUMENTATION:
 * - GPU functions: Use cudaEvent or clock64() for per-thread timing
 * - CPU functions: Use std::chrono::high_resolution_clock
 * - Target consumer functions marked with "TIMING POINT" in documentation
 * 
 * See REFERENCE.md for complete documentation of all functions and data structures.
 * 
 * @author Soham Bagchi
 */

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

#include "types.h"

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
 

/**
 * @brief Query GPU device properties and return clock rate
 * @return GPU clock rate in KHz
 * @note Other device properties are queried but output is commented out
 */
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


/**
 * @brief Background GPU writer - no synchronization, just keeps GPU busy
 * @param buffer Buffer to write to
 * 
 * NO TIMING NEEDED - This is background load only
 * Used to create concurrent activity during propagation tests
 */
__device__ static void __attribute__((optimize("O0"))) gpu_dummy_writer_worker(bufferElement *buffer) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(i, cuda::memory_order_relaxed);
        }
    }
}

/**
 * @brief Background GPU reader - no synchronization, just keeps GPU busy
 * @param buffer Buffer to read from
 * @param results Per-thread result storage
 * 
 * NO TIMING NEEDED - This is background load only
 * Used to create concurrent activity during propagation tests
 */
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

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_writer_worker_propagation(bufferElement *buffer, flag_d *r_signal) {

    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        if (i % (NUM_ITERATIONS / 4) == 0) {
            cudaSleep(10000000000);
        }
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(i, cuda::memory_order_relaxed);
        }
    }
}

__device__ static void __attribute__((optimize("O0"))) gpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, flag_d *r_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint result = 0;

    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result += buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }

    results[tid].data = result;
}

/**
 * @brief GPU writer for propagation hierarchy testing (simple variant)
 * @param buffer Buffer to write data to
 * @param r_signal Reader readiness signal (waits for all readers)
 * @param w_signal Writer completion signal (signals when done)
 * 
 * TIMING POINT: Time the write loop and flag store operation
 * Writes value 1 to all buffer elements, then sets w_signal to 1
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation(bufferElement *buffer, flag_d *r_signal, flag_d *w_signal) {

    while(r_signal->flag.load(cuda::memory_order_acquire) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }
    
    // for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            buffer[j].data.store(10, cuda::memory_order_relaxed);
        }
    // }

    // Set Writer Signal
    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
}

/**
 * @brief GPU reader for propagation testing
 * @param buffer Buffer to read from
 * @param results Per-thread result storage
 * @param r_signal Reader readiness counter (increments to signal ready)
 * @param w_signal Writer completion flag (waits with C_H_FLAG_LOAD_ORDER)
 * 
 * TIMING POINT: Start when w_signal observed non-zero, end after buffer read
 * Uses C_H_FLAG_LOAD_ORDER (acquire/relaxed) on flag load, relaxed for buffer data
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation(bufferElement *buffer, bufferElement_na *results, flag_d *r_signal, flag_d *w_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;

    
    // Set Reader Signal
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    result = 0;
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result;
}

template <typename B, typename W, typename R>
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_reader_propagation_hierarchy(B *buffer, bufferElement_na *results, R * r_signal, flag_t * w_t_signal, flag_b * w_b_signal, flag_d * w_d_signal, flag_s * w_s_signal, flag_s * fallback_signal) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint init_flag_t = w_t_signal->flag.load(cuda::memory_order_relaxed);
    uint init_flag_b = w_b_signal->flag.load(cuda::memory_order_relaxed);
    uint init_flag_d = w_d_signal->flag.load(cuda::memory_order_relaxed);
    uint init_flag_s = w_s_signal->flag.load(cuda::memory_order_relaxed);

    uint result = 0;

    #ifdef CONSUMERS_CACHE
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    #endif

    results[tid].data = result;

    // Set Reader Signal
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while ((init_flag_t = w_t_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_b = w_b_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_d = w_d_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_s = w_s_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[tid].data = result + init_flag_t * 1000000000 + init_flag_b * 100000000 + init_flag_d * 10000000 + init_flag_s * 1000000;

    printf("B[%d] T[%d] (%d:%d:%d) Result %d\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4);

    // cudaSleep(10000000000);

    // printf("B[%d] T[%d] (%d:%d:%d) Done\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4);
}

template <typename B, typename W, typename R>
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_propagation_hierarchy(B *buffer, bufferElement_na *results, R *r_signal, W *w_signal, flag_s *fallback_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;

    #ifdef CONSUMERS_CACHE
    uint init_flag = w_signal->flag.load(cuda::memory_order_relaxed);

    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    #endif
    
    // w_signal->flag.store(, cuda::memory_order_relaxed);
    results[tid].data = result;
    
    // Set Reader Signal
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    result = 0;
    
    while(w_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0 && fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait for Writer Signal
    }

    // cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_device);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    printf("B[%d] T[%d] (%d:%d:%d) Result %d\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4, results[tid].data);

    // cudaSleep(10000000000);

    // printf("B[%d] T[%d] (%d:%d:%d) Done\n", blockIdx.x, threadIdx.x, threadIdx.x / 32, threadIdx.x % 8, threadIdx.x % 4);
}

/**
 * @brief GPU writer for propagation hierarchy testing (homogeneous GPU-only)
 * @param buffer Main data buffer
 * @param r_signal Reader readiness counter (waits for GPU_NUM_THREADS * GPU_NUM_BLOCKS - 1)
 * @param w_t_signal Writer flag for thread scope
 * @param w_b_signal Writer flag for block scope
 * @param w_d_signal Writer flag for device scope
 * @param w_s_signal Writer flag for system scope
 * @param fallback_signal Timeout mechanism
 * 
 * TIMING POINTS: Time each phase separately:
 *   Phase 1: Write 10 to buffer + set w_t_signal → cudaSleep(10000000000)
 *   Phase 2: Write 20 to buffer + set w_b_signal → sleep
 *   Phase 3: Write 30 to buffer + set w_d_signal → sleep
 *   Phase 4: Write 40 to buffer + set w_s_signal → sleep
 * 
 * This is the MAIN WRITER for single-writer propagation hierarchy tests
 * Flag store order controlled by P_H_FLAG_STORE_ORDER compile flag
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation_hierarchy(bufferElement *buffer, flag_d * r_signal, flag_t * w_t_signal, flag_b * w_b_signal, flag_d * w_d_signal, flag_s * w_s_signal, flag_s * fallback_signal) {

    printf("GPU Writer (Block: %d, Core: %d, Total: %d) (Waiting For: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x, GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Thread)
    w_t_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Block)
    w_b_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Device)
    w_d_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (System)

    w_s_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    cudaSleep(10000000000);
    
    fallback_signal->flag.store(4, cuda::memory_order_release);

    cudaSleep(10000000000);

    printf("[GPU] Writer Done\n");
}

/**
 * @brief GPU writer for propagation hierarchy testing (heterogeneous CPU+GPU)
 * @param buffer Main data buffer
 * @param r_signal Reader readiness counter (waits for CPU_NUM_THREADS + GPU_NUM_THREADS - 1)
 * @param w_t_signal Writer flag for thread scope
 * @param w_b_signal Writer flag for block scope
 * @param w_d_signal Writer flag for device scope
 * @param w_s_signal Writer flag for system scope
 * @param fallback_signal Timeout mechanism
 * 
 * TIMING POINTS: Same as gpu_buffer_writer_propagation_hierarchy
 *   Four phases, each: write value to buffer + set scope flag + sleep
 * 
 * This writer expects both CPU and GPU readers
 * Flag store order controlled by P_H_FLAG_STORE_ORDER compile flag
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_writer_propagation_hierarchy_cpu(bufferElement *buffer, flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal) {
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("GPU Het-Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Thread)
    w_t_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Block)
    w_b_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (Device)
    w_d_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    // Set Writer Signals (System)
    w_s_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    cudaSleep(10000000000);

    fallback_signal->flag.store(4, cuda::memory_order_release);

    // cudaSleep(10000000000);

    // printf("[GPU] Het-Writer Done\n");
}


// TODO: GPU Spawner
/**
 * @brief Orchestrator kernel for propagation hierarchy testing
 * @param buffer Main data buffer
 * @param w_buffer Writer buffer (for divergent execution)
 * @param results Per-thread result storage
 * @param r_signal Reader readiness counter
 * @param w_t_signal Thread scope writer flag
 * @param w_b_signal Block scope writer flag
 * @param w_d_signal Device scope writer flag
 * @param w_s_signal System scope writer flag
 * @param fallback_signal Timeout mechanism
 * @param spawn_writer Control flag (CE_WRITER, CE_HET_WRITER, CE_NO_WRITER)
 * 
 * Thread role assignment:
 *   Block 0, Thread 0: Writer (if spawn_writer != CE_NO_WRITER)
 *   Other threads: Readers based on threadIdx.x % 8:
 *     0-3: Relaxed readers (thread/block/device/system scopes)
 *     4-7: Acquire readers (or dummy readers if NO_ACQ defined)
 * 
 * NO TIMING NEEDED - Individual consumer functions handle their own timing
 */
__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_writer_propagation_hierarchy(bufferElement *buffer, bufferElement *w_buffer, bufferElement_na * results, flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal, WriterType *spawn_writer) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    
    if ((*spawn_writer == CE_WRITER || *spawn_writer == CE_HET_WRITER) && blockId == 0 && threadId == 0) {
        // gpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        if (*spawn_writer == CE_WRITER) {
            gpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            gpu_buffer_writer_propagation_hierarchy_cpu(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else if (blockId == 0) {
        if (threadId % 8 == 0) {
            gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (threadId % 8 == 1) {
            gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (threadId % 8 == 2) {
            gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (threadId % 8 == 3) {
            gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_s_signal, fallback_signal);
        } else if (threadId % 8 == 4) {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        } else if (threadId % 8 == 5) {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        } else if (threadId % 8 == 6) {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        } else if (threadId % 8 == 7) {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        }
    } else {
        if (threadId == 0) {
            gpu_dummy_writer_worker_propagation(w_buffer, r_signal);
        } else if (threadId < 32) {
            if (threadId % 8 == 0) {
                gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_t_signal, fallback_signal);
            } else if (threadId % 8 == 1) {
                gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_b_signal, fallback_signal);
            } else if (threadId % 8 == 2) {
                gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_d_signal, fallback_signal);
            } else if (threadId % 8 == 3) {
                gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_s_signal, fallback_signal);
            } else if (threadId % 8 == 4) {
                // gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_t_signal, fallback_signal);
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            } else if (threadId % 8 == 5) {
                // gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_b_signal, fallback_signal);
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            } else if (threadId % 8 == 6) {
                // gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_d_signal, fallback_signal);
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            } else if (threadId % 8 == 7) {
                // gpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_s_signal, fallback_signal);
                gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            }
        } else {
            gpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
        }
    }
} 


__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_thread_propagation_hierarchy_cpu(bufferElement_t * buffer, flag_d * r_signal, flag_t * w_signal, flag_s * fb_signal) {

    printf("GPU Thread Het-Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Thread Het-Writer Done\n");
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_block_propagation_hierarchy_cpu(bufferElement_b * buffer, flag_d * r_signal, flag_b * w_signal, flag_s * fb_signal) {

    printf("GPU Block Het-Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Block Het-Writer Done\n");
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_device_propagation_hierarchy_cpu(bufferElement_d * buffer, flag_d * r_signal, flag_d * w_signal, flag_s * fb_signal) {

    printf("GPU Device Het-Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Device Het-Writer Done\n");
}

__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_system_propagation_hierarchy_cpu(bufferElement_s * buffer, flag_d * r_signal, flag_s * w_signal, flag_s * fb_signal) {

    printf("GPU System Het-Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU System Het-Writer Done\n");
}
/**
 * @brief GPU multi-writer for THREAD scope (homogeneous GPU-only)
 * @param buffer Thread-scoped atomic buffer
 * @param r_signal Reader readiness counter (waits for GPU_NUM_THREADS * GPU_NUM_BLOCKS - 4)
 * @param w_signal Thread scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * 
 * TIMING POINT: Time write loop (value 10) + flag store → sleep → write (value 1)
 * Part of multi-writer pattern with 4 concurrent writers
 * Flag store uses P_H_FLAG_STORE_ORDER
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_thread_propagation_hierarchy(bufferElement_t * buffer, flag_d * r_signal, flag_t * w_signal, flag_s * fb_signal) {

    printf("GPU Thread Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Thread Writer Done\n");
}

/**
 * @brief GPU multi-writer for BLOCK scope (homogeneous GPU-only)
 * @param buffer Block-scoped atomic buffer
 * @param r_signal Reader readiness counter (waits for GPU_NUM_THREADS * GPU_NUM_BLOCKS - 4)
 * @param w_signal Block scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * 
 * TIMING POINT: Time write loop (value 20) + flag store → sleep → write (value 2)
 * Part of multi-writer pattern with 4 concurrent writers
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_block_propagation_hierarchy(bufferElement_b * buffer, flag_d * r_signal, flag_b * w_signal, flag_s * fb_signal) {

    printf("GPU Block Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Block Writer Done\n");
}

/**
 * @brief GPU multi-writer for DEVICE scope (homogeneous GPU-only)
 * @param buffer Device-scoped atomic buffer
 * @param r_signal Reader readiness counter (waits for GPU_NUM_THREADS * GPU_NUM_BLOCKS - 4)
 * @param w_signal Device scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * 
 * TIMING POINT: Time write loop (value 30) + flag store → sleep → write (value 3)
 * Part of multi-writer pattern with 4 concurrent writers
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_device_propagation_hierarchy(bufferElement_d * buffer, flag_d * r_signal, flag_d * w_signal, flag_s * fb_signal) {

    printf("GPU Device Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU Device Writer Done\n");
}

/**
 * @brief GPU multi-writer for SYSTEM scope (homogeneous GPU-only)
 * @param buffer System-scoped atomic buffer
 * @param r_signal Reader readiness counter (waits for GPU_NUM_THREADS * GPU_NUM_BLOCKS - 4)
 * @param w_signal System scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * 
 * TIMING POINT: Time write loop (value 40) + flag store → sleep → write (value 4)
 * Part of multi-writer pattern with 4 concurrent writers
 */
__device__ static void __attribute__((optimize("O0"))) gpu_buffer_multi_writer_system_propagation_hierarchy(bufferElement_s * buffer, flag_d * r_signal, flag_s * w_signal, flag_s * fb_signal) {

    printf("GPU System Writer (Block: %d, Core: %d, Total: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);

    uint result = 0; 

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    // cudaSleep(10000000000);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    cudaSleep(10000000000);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    // cudaSleep(10000000000);

    // printf("GPU System Writer Done\n");
}

// TODO: GPU Spawner
/**
 * @brief Orchestrator kernel for multi-writer propagation hierarchy testing
 * @param dummy_buffer Background load buffer
 * @param buffer_t Thread-scoped atomic buffer
 * @param buffer_b Block-scoped atomic buffer
 * @param buffer_d Device-scoped atomic buffer
 * @param buffer_s System-scoped atomic buffer
 * @param results Per-thread result storage
 * @param r_signal Reader readiness counter
 * @param w_signal_t Thread scope writer flag
 * @param w_signal_b Block scope writer flag
 * @param w_signal_d Device scope writer flag
 * @param w_signal_s System scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * @param spawn_writer Control flag (CE_WRITER, CE_HET_WRITER, CE_NO_WRITER)
 * 
 * Thread role assignment:
 *   Blocks 0-3, Thread 0: Four writers (one per scope level)
 *   Other threads: Readers based on global_tid % 8
 * 
 * NO TIMING NEEDED - Individual consumer functions handle their own timing
 */
__global__ static void __attribute__((optimize("O0"))) gpu_buffer_reader_multi_writer_propagation_hierarchy(bufferElement * dummy_buffer, bufferElement_t * buffer_t, bufferElement_b * buffer_b, bufferElement_d * buffer_d, bufferElement_s * buffer_s, bufferElement_na * results, flag_d * r_signal, flag_t * w_signal_t, flag_b * w_signal_b, flag_d * w_signal_d, flag_s * w_signal_s,  flag_s * fb_signal, WriterType *spawn_writer) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (*spawn_writer != CE_NO_WRITER && tid == 0 && bid >= 0 && bid < 4) {
        switch (bid) {
            case 0: 
                if (*spawn_writer == CE_HET_WRITER) {
                    gpu_buffer_multi_writer_thread_propagation_hierarchy_cpu(buffer_t, r_signal, w_signal_t, fb_signal);
                } else {
                    gpu_buffer_multi_writer_thread_propagation_hierarchy(buffer_t, r_signal, w_signal_t, fb_signal);
                }
                break;
            case 1:
                if (*spawn_writer == CE_HET_WRITER) {
                    gpu_buffer_multi_writer_block_propagation_hierarchy_cpu(buffer_b, r_signal, w_signal_b, fb_signal);
                } else {
                    gpu_buffer_multi_writer_block_propagation_hierarchy(buffer_b, r_signal, w_signal_b, fb_signal);
                }
                break;
            case 2:
                if (*spawn_writer == CE_HET_WRITER) {
                    gpu_buffer_multi_writer_device_propagation_hierarchy_cpu(buffer_d, r_signal, w_signal_d, fb_signal);
                } else {
                    gpu_buffer_multi_writer_device_propagation_hierarchy(buffer_d, r_signal, w_signal_d, fb_signal);
                }
                break;
            case 3:
                if (*spawn_writer == CE_HET_WRITER) {
                    gpu_buffer_multi_writer_system_propagation_hierarchy_cpu(buffer_s, r_signal, w_signal_s, fb_signal);
                } else {
                    gpu_buffer_multi_writer_system_propagation_hierarchy(buffer_s, r_signal, w_signal_s, fb_signal);
                }
                break;
            default:
                break;
        }
    } else {
        switch (global_tid % 8) {
            case 0:
                gpu_buffer_reader_propagation_hierarchy(buffer_t, results, r_signal, w_signal_t, fb_signal);
                break;
            case 1:
                gpu_buffer_reader_propagation_hierarchy(buffer_b, results, r_signal, w_signal_b, fb_signal);
                break;
            case 2:
                gpu_buffer_reader_propagation_hierarchy(buffer_d, results, r_signal, w_signal_d, fb_signal);
                break;
            case 3:
                gpu_buffer_reader_propagation_hierarchy(buffer_s, results, r_signal, w_signal_s, fb_signal);
                break;
            case 4:
                gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 5:
                gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 6:
                gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 7:
                gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            default:
                break;
        }
    }
}

