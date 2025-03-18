#ifndef CACHE_INVALIDATION_TESTING_PROP_RADIUS_CUH
#define CACHE_INVALIDATION_TESTING_PROP_RADIUS_CUH

#include "cache_invalidation_testing_utils.cuh"

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

#endif