#ifndef CACHE_INVALIDATION_TESTING_LOOPING_CUH
#define CACHE_INVALIDATION_TESTING_LOOPING_CUH

#include "cache_invalidation_testing_utils.cuh"

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

#endif