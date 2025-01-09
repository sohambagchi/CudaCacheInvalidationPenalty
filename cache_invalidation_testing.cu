// - needs memory allocator selection support
// - number of iterations
// - some frequency metric?

#include <cuda/atomic>
#include <iostream>
#include <chrono>
#include <fstream>
#include <thread>
#include <numa.h>
#include <unistd.h>
#include <string.h>
#include <vector>


#define BUFFER_SIZE 32768
#define NUM_ITERATIONS 2000000

#ifndef DATA_SIZE
#define DATA_SIZE uint64_t
#endif

#ifndef CUDA_THREAD_SCOPE
#define CUDA_THREAD_SCOPE cuda::thread_scope_system
#endif

typedef enum {
    CE_SYS_MALLOC,
    CE_NUMA_HOST,
    CE_NUMA_DEVICE,
    CE_DRAM,
    CE_UM
} AllocatorType;

// #pragma GCC push_options
// #pragma GCC optimize ("O0")

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

// #pragma GCC pop_options

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
//         // buffer[i].store(num, cuda::memory_order_release);
//         buffer[i] = num;
//     }
// }

// write a function that prints out a bunch of gpu properties (whatever is available from the cuda API), and then return the frequency of the GPU

int get_gpu_frequency() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << std::endl;
    std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Kernel Execution Timeout Enabled: " << prop.kernelExecTimeoutEnabled << std::endl;
    std::cout << "Integrated: " << prop.integrated << std::endl;
    std::cout << "Can Map Host Memory: " << prop.canMapHostMemory << std::endl;
    std::cout << "Compute Mode: " << prop.computeMode << std::endl;
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
    std::cout << "ECC Enabled: " << prop.ECCEnabled << std::endl;
    std::cout << "PCI Bus ID: " << prop.pciBusID << std::endl;
    std::cout << "PCI Device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "TCC Driver: " << prop.tccDriver << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize << std::endl;
    std::cout << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Stream Priorities Supported: " << prop.streamPrioritiesSupported << std::endl;
    std::cout << "Global L1 Cache Supported: " << prop.globalL1CacheSupported << std::endl;
    std::cout << "Local L1 Cache Supported: " << prop.localL1CacheSupported << std::endl;
    std::cout << "Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "Managed Memory: " << prop.managedMemory << std::endl;
    std::cout << "Is Multi-GPU Board: " << prop.isMultiGpuBoard << std::endl;
    std::cout << "Multi-GPU Board Group ID: " << prop.multiGpuBoardGroupID << std::endl;

    
    return prop.clockRate;
}

long long duration_in_nanoseconds(unsigned long duration, int gpu_clock_rate) {
    return (long long) duration * 1000000000 / gpu_clock_rate;
}


int main(int argc, char* argv[]) {
    // parse arguments

    if (argc < 2) {
        std::cout << "Error: No arguments provided" << std::endl;
        return 0;
    }


    AllocatorType allocator_t;
    // DATA_SIZE NUM_ITERATIONS_t;
    // DATA_SIZE frequency_metric_t;

    int opt;

    while ((opt = getopt(argc, argv, "m:")) != -1) {
        switch (opt) {
            case 'm':
                if (strcmp(optarg, "malloc") == 0) {
                    allocator_t = CE_SYS_MALLOC;
                } else if (strcmp(optarg, "numa_host") == 0) {
                    allocator_t = CE_NUMA_HOST;
                } else if (strcmp(optarg, "numa_device") == 0) {
                    allocator_t = CE_NUMA_DEVICE;
                } else if (strcmp(optarg, "dram") == 0) {
                    allocator_t = CE_DRAM;
                } else if (strcmp(optarg, "um") == 0) {
                    allocator_t = CE_UM;
                } else {
                    std::cout << "Error: Invalid memory allocator" << std::endl;
                    return 0;
                }
                std::cout << "[INFO] Memory allocator: " << optarg << std::endl;
                break;
            // case 'f':
            //     frequency_metric_t = atoi(optarg);    
            //     std::cout << "Frequency metric: " << optarg << std::endl;
            //     break;
            default:
                std::cout << "Error: Invalid argument" << std::endl;
                return 0;
        }
    }

    const AllocatorType allocator = allocator_t;
    // const DATA_SIZE NUM_ITERATIONS = NUM_ITERATIONS_t;
    // const DATA_SIZE frequency_metric = frequency_metric_t;

    // if (frequency_metric > NUM_ITERATIONS) {
    //     std::cout << "Error: Frequency metric cannot be greater than number of iterations" << std::endl;
    //     return 0;
    // }

    // int durations[NUM_ITERATIONS];
    // DATA_SIZE * durations = (DATA_SIZE *) malloc(sizeof(DATA_SIZE) * NUM_ITERATIONS);

    // for (int i = 0; i < NUM_ITERATIONS; i++) {
    //     durations[i] = 0;
    // }

    cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer;
    // DATA_SIZE*buffer;

    if (allocator == CE_SYS_MALLOC) {
        std::cout << "[INFO] Allocating memory using sys malloc" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) malloc(sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE);
        // buffer = (DATA_SIZE*) malloc(sizeof(DATA_SIZE) * BUFFER_SIZE);
    } else if (allocator == CE_NUMA_HOST) {
        std::cout << "[INFO] Allocating memory using numa host" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) numa_alloc_onnode(sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE, 0);
        // buffer = (DATA_SIZE *) numa_alloc_onnode(sizeof(DATA_SIZE) * BUFFER_SIZE, 0);
    } else if (allocator == CE_NUMA_DEVICE) {
        std::cout << "[INFO] Allocating memory using numa device" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) numa_alloc_onnode(sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE, 1);
        // buffer = (DATA_SIZE *) numa_alloc_onnode(sizeof(DATA_SIZE) * BUFFER_SIZE, 1);
    } else if (allocator == CE_DRAM) {
        std::cout << "[INFO] Allocating memory using cudaMallocHost" << std::endl;
        cudaMallocHost((void **) &buffer, sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE);
        // cudaMallocHost((void **) &buffer, sizeof(DATA_SIZE) * BUFFER_SIZE);
    } else if (allocator == CE_UM) {
        std::cout << "[INFO] Allocating memory using Unified Memory" << std::endl;
        cudaMallocManaged((void **) &buffer, sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE);
        // cudaMallocManaged((void **) &buffer, sizeof(DATA_SIZE) * BUFFER_SIZE);
    }

    

    std::cout << "[INFO] Size of Array: " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE / 1024 / 1024 << "MB" << std::endl;

    // std::chrono::high_resolution_clock::time_point kernel_begin = std::chrono::high_resolution_clock::now();
    // gpuTrigger<<<1,1>>>((cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) buffer, 0);
    // cudaDeviceSynchronize();
    // std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    // int kernel_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin).count();
    // std::cout << "Kernel duration: " << kernel_duration << std::endl;
    std::chrono::high_resolution_clock::time_point store_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].store(1, cuda::memory_order_relaxed);
        // buffer[i] = 1;
    }
    std::chrono::high_resolution_clock::time_point store_end = std::chrono::high_resolution_clock::now();
    int store_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(store_end - store_begin).count();
    std::cout << "[INFO] Baseline Store duration: " << store_duration << std::endl;

    // int local_buffer;
    
    // std::chrono::high_resolution_clock::time_point load_begin = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < BUFFER_SIZE; i++) {
    //     // local_buffer += ((std::atomic<DATA_SIZE>) buffer[i]).load();
    //     local_buffer += buffer[i].load(cuda::memory_order_relaxed);
    // }
    // std::chrono::high_resolution_clock::time_point load_end = std::chrono::high_resolution_clock::now();
    // int load_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_begin).count();
    // std::cout << "Baseline Load duration: " << load_duration << std::endl;

    // std::cout << "buffer sum: " << local_buffer << std::endl;

    // std::chrono::high_resolution_clock::time_point load_begin_ = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < BUFFER_SIZE; i++) {
    //     local_buffer[i] = ((std::atomic<DATA_SIZE>) buffer[i]).load(std::memory_order_relaxed);
    // }
    // std::chrono::high_resolution_clock::time_point load_end_ = std::chrono::high_resolution_clock::now();
    // int load_duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end_ - load_begin_).count();
    // std::cout << "Baseline Load duration: " << load_duration_ << std::endl;



    // free(local_buffer);


    std::cout << "[INFO] Starting buffer reader" << std::endl;
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::thread single_iter_cpu(buffer_reader_single_iter, buffer);
    single_iter_cpu.join();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Buffer reader completed" << std::endl;

    int duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    std::cout << "[TIME] Duration: " << duration << std::endl;


    // std::chrono::high_resolution_clock::time_point kernel_begin = std::chrono::high_resolution_clock::now();

    const int chunkSize = 4194304 * 16;
    const int threadsPerBlock = 16;
    const int totalThreads = (BUFFER_SIZE + chunkSize - 1) / chunkSize;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    const int gpu_clock_rate = get_gpu_frequency();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // gpuTrigger<<<blocksPerGrid,threadsPerBlock>>>(buffer, 1, chunkSize);
    // // gpuTrigger<<<1,1>>>((cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) buffer, 0);
    // // std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // // int kernel_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin).count();
    
    
    // std::cout << "[TIME] Kernel duration: " << kernel_duration << std::endl;
    
    
    std::thread cpu(buffer_reader, buffer);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(cpu.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    
    // DATA_SIZE epoch = load_duration * NUM_ITERATIONS / frequency_metric;
    
    std::ofstream durations_file;
    int n = 0;
    
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        
        // auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "[GPU] ------------------------------- Triggering Kernel" << std::endl;
        cudaEventRecord(start);
        // gpuTrigger<<<blocksPerGrid, threadsPerBlock>>>(buffer, n, chunkSize);
        gpuTrigger<<<1, 1>>>(buffer, n, BUFFER_SIZE);
        cudaEventRecord(stop);
        n++;
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        float kernel_duration;
        cudaEventElapsedTime(&kernel_duration, start, stop);
        
        auto end = std::chrono::high_resolution_clock::now();
        // unsigned long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "[GPU] ------------------------------- Kernel Completed " << kernel_duration * 1000000 << std::endl; //duration_in_nanoseconds(duration, gpu_clock_rate) << std::endl;
    }
    
    return;
    cpu.join();

    // output all the duration values to a txt durations_file;, one on each line


    free(buffer);

    // free(local_buffer);

    // free memory




    // allocate memory
    // run kernel
    // free memory
    // print results
    return 0;
}
