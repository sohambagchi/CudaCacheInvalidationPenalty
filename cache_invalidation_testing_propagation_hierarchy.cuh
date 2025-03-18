#ifndef CACHE_INVALIDATION_TESTING_PROP_HIERARCHY_CUH
#define CACHE_INVALIDATION_TESTING_PROP_HIERARCHY_CUH

#include "cache_invalidation_testing_utils.cuh"


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



#endif