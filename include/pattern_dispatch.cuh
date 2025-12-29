#ifndef PATTERN_DISPATCH_CUH
#define PATTERN_DISPATCH_CUH

#include "pattern_config.hpp"
#include "types.hpp"
#include <cuda/atomic>

// Device sleep function for synchronization delays
__device__ void cudaSleep(clock_t sleep_cycles) {
    clock_t start = clock64();
    clock_t cycles_elapsed;
    do { 
        cycles_elapsed = clock64() - start; 
    } while (cycles_elapsed < sleep_cycles);
}

// Device-side constant memory for pattern configuration
// Note: Will be populated via cudaMemcpyToSymbol from host
__constant__ ThreadConfig d_pattern_gpu[8][64];

// ============================================================================
// TIMING CALIBRATION
// ============================================================================

/**
 * @brief Calibration kernel to measure clock64() overhead
 * Performs repeated clock64() calls to measure average overhead
 */
__global__ void calibrate_clock_overhead(clock_t *results, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        clock_t total_overhead = 0;
        for (int i = 0; i < num_samples; i++) {
            clock_t start = clock64();
            clock_t end = clock64();
            total_overhead += (end - start);
        }
        results[0] = total_overhead / num_samples;
    }
}

// ============================================================================
// GPU READER VARIANTS
// ============================================================================

/**
 * @brief GPU buffer reader with ACQUIRE memory ordering
 * Waits for writer signal with acquire semantics
 */
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_acquire_no_cache(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    timing[tid].block_id = blockIdx.x;
    timing[tid].thread_id = threadIdx.x;
    timing[tid].consumer_type = 1;  // reader_acq
    timing[tid].caching = false;
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    timing[tid].start_time = clock64();

    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    timing[tid].flag_trigger_time = clock64();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    timing[tid].end_time = clock64();
    
    printf("B[%d] T[%d] NC-ACQ Result %d\n", blockIdx.x, threadIdx.x, result);
}

/**
 * @brief GPU buffer reader with ACQUIRE memory ordering
 * Waits for writer signal with acquire semantics
 */
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_acquire_caching(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    timing[tid].block_id = blockIdx.x;
    timing[tid].thread_id = threadIdx.x;
    timing[tid].consumer_type = 1;  // reader_acq
    timing[tid].caching = true;
    
    // Prefetch buffer to warm cache BEFORE signaling readiness
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    timing[tid].start_time = clock64();

    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    timing[tid].flag_trigger_time = clock64();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    timing[tid].end_time = clock64();
    
    printf("B[%d] T[%d] C-ACQ Result %d\n", blockIdx.x, threadIdx.x, result);
}

/**
 * @brief GPU buffer reader with RELAXED memory ordering
 * Waits for writer signal with relaxed semantics
 */
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_relaxed_no_cache(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    timing[tid].block_id = blockIdx.x;
    timing[tid].thread_id = threadIdx.x;
    timing[tid].consumer_type = 2;  // reader_rlx
    timing[tid].caching = false;
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;

    // Record start time
    timing[tid].start_time = clock64();
    
    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    timing[tid].flag_trigger_time = clock64();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    timing[tid].end_time = clock64();
    
    printf("B[%d] T[%d] NC-RLX Result %d\n", blockIdx.x, threadIdx.x, result);
}

/**
 * @brief GPU buffer reader with RELAXED memory ordering
 * Waits for writer signal with relaxed semantics
 */
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_relaxed_caching(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    timing[tid].block_id = blockIdx.x;
    timing[tid].thread_id = threadIdx.x;
    timing[tid].consumer_type = 2;  // reader_rlx
    timing[tid].caching = true;
    
    // Prefetch buffer to warm cache BEFORE signaling readiness
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;

    // Record start time
    timing[tid].start_time = clock64();
    
    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    timing[tid].flag_trigger_time = clock64();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    timing[tid].end_time = clock64();
    
    printf("B[%d] T[%d] C-RLX Result %d\n", blockIdx.x, threadIdx.x, result);
}

// ============================================================================
// GPU WRITER VARIANTS
// ============================================================================

/**
 * @brief GPU buffer writer with RELEASE memory ordering
 * Sets flags with release semantics
 */
__device__ void gpu_buffer_writer_release(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("GPU Writer (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    fallback_signal->flag.store(4, cuda::memory_order_release);
}

/**
 * @brief GPU buffer writer with RELAXED memory ordering
 * Sets flags with relaxed semantics
 */
__device__ void gpu_buffer_writer_relaxed(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("GPU Writer (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    fallback_signal->flag.store(4, cuda::memory_order_release);
}

// ============================================================================
// GPU MULTI-WRITER VARIANTS
// ============================================================================

/**
 * @brief Multi-writer thread-scope with RELEASE ordering
 */
__device__ void gpu_buffer_multi_writer_thread_release(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer THREAD (Release) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer thread-scope with RELAXED ordering
 */
__device__ void gpu_buffer_multi_writer_thread_relaxed(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer THREAD (Relaxed) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer block-scope with RELEASE ordering
 */
__device__ void gpu_buffer_multi_writer_block_release(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer BLOCK (Release) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer block-scope with RELAXED ordering
 */
__device__ void gpu_buffer_multi_writer_block_relaxed(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer BLOCK (Relaxed) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer device-scope with RELEASE ordering
 */
__device__ void gpu_buffer_multi_writer_device_release(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer DEVICE (Release) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer device-scope with RELAXED ordering
 */
__device__ void gpu_buffer_multi_writer_device_relaxed(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer DEVICE (Relaxed) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer system-scope with RELEASE ordering
 */
__device__ void gpu_buffer_multi_writer_system_release(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer SYSTEM (Release) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer system-scope with RELAXED ordering
 */
__device__ void gpu_buffer_multi_writer_system_relaxed(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
) {
    printf("GPU Multi-Writer SYSTEM (Relaxed) B[%d] T[%d]\n", blockIdx.x, threadIdx.x);
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

// ============================================================================
// GPU DISPATCH LOGIC
// ============================================================================

/**
 * @brief Helper to convert ThreadScope to uint8_t for timing data
 */
__device__ inline uint8_t scope_to_uint8(ThreadScope scope) {
    switch (scope) {
        case ThreadScope::THREAD: return 0;
        case ThreadScope::BLOCK: return 1;
        case ThreadScope::DEVICE: return 2;
        case ThreadScope::SYSTEM: return 3;
        default: return 2;  // Default to device
    }
}

/**
 * @brief Dispatch reader to appropriate variant based on ordering
 */
__device__ void dispatch_reader(
    ThreadConfig cfg,
    bufferElement *buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    // Set flag type in timing data
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    timing[tid].flag_type = scope_to_uint8(cfg.watch_flag);
    
    if (cfg.ordering == MemoryOrdering::ACQUIRE) {
        // Dispatch to acquire reader with appropriate flag type
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                break;
        }
    } else {  // RELAXED
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                break;
        }
    }
}

/**
 * @brief Dispatch writer to appropriate variant based on ordering
 */
__device__ void dispatch_writer(
    ThreadConfig cfg,
    bufferElement *buffer,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    if (cfg.ordering == MemoryOrdering::RELEASE) {
        gpu_buffer_writer_release(buffer, r_signal,
                                 w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                 fallback_signal);
    } else {  // RELAXED
        gpu_buffer_writer_relaxed(buffer, r_signal,
                                 w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                 fallback_signal);
    }
}

/**
 * @brief Dispatch multi-writer to appropriate scope-specific variant
 */
__device__ void dispatch_multi_writer(
    ThreadConfig cfg,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    bool is_release = (cfg.ordering == MemoryOrdering::RELEASE);
    
    switch (cfg.scope) {
        case ThreadScope::THREAD:
            if (is_release)
                gpu_buffer_multi_writer_thread_release(buffer_t, r_signal, w_t_signal, fallback_signal);
            else
                gpu_buffer_multi_writer_thread_relaxed(buffer_t, r_signal, w_t_signal, fallback_signal);
            break;
        case ThreadScope::BLOCK:
            if (is_release)
                gpu_buffer_multi_writer_block_release(buffer_b, r_signal, w_b_signal, fallback_signal);
            else
                gpu_buffer_multi_writer_block_relaxed(buffer_b, r_signal, w_b_signal, fallback_signal);
            break;
        case ThreadScope::DEVICE:
            if (is_release)
                gpu_buffer_multi_writer_device_release(buffer_d, r_signal, w_d_signal, fallback_signal);
            else
                gpu_buffer_multi_writer_device_relaxed(buffer_d, r_signal, w_d_signal, fallback_signal);
            break;
        case ThreadScope::SYSTEM:
            if (is_release)
                gpu_buffer_multi_writer_system_release(buffer_s, r_signal, w_s_signal, fallback_signal);
            else
                gpu_buffer_multi_writer_system_relaxed(buffer_s, r_signal, w_s_signal, fallback_signal);
            break;
    }
}

/**
 * @brief Dispatch multi-reader to scope-matching buffer
 */
__device__ void dispatch_multi_reader(
    ThreadConfig cfg,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    // Set flag type in timing data
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    timing[tid].flag_type = scope_to_uint8(cfg.watch_flag);
    
    if (cfg.ordering == MemoryOrdering::ACQUIRE) {
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer_t, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer_b, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer_d, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching)
                    gpu_buffer_reader_acquire_caching(buffer_s, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_acquire_no_cache(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                break;
        }
    } else {  // RELAXED
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer_t, results, r_signal,
                                        w_t_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer_b, results, r_signal,
                                        w_b_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer_d, results, r_signal,
                                        w_d_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching)
                    gpu_buffer_reader_relaxed_caching(buffer_s, results, r_signal,
                                        w_s_signal, fallback_signal, timing);
                else
                    gpu_buffer_reader_relaxed_no_cache(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                break;
        }
    }
}

// ============================================================================
// DUMMY FUNCTIONS FOR BACKGROUND LOAD
// ============================================================================

/**
 * @brief Dummy GPU reader for background load
 * Performs reads without synchronization to create memory traffic
 */
__device__ void gpu_dummy_reader_worker_propagation(
    bufferElement *dummy_buffer,
    bufferElement_na *results, 
    flag_d *r_signal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    // Signal readiness
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    // Background read load
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            result += dummy_buffer[i].data.load(cuda::memory_order_relaxed);
        }
    }
    
    results[tid].data = result;
}

/**
 * @brief Dummy GPU writer for background load
 * Performs writes without synchronization to create memory traffic
 */
__device__ void gpu_dummy_writer_worker_propagation(
    bufferElement *dummy_buffer, 
    flag_d *r_signal
) {
    // Wait for readers to be ready
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    // Background write load
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            dummy_buffer[i].data.store(iter + i, cuda::memory_order_relaxed);
        }
    }
}

// ============================================================================
// THREAD DISPATCH FUNCTIONS
// ============================================================================

// Forward declarations removed (functions now defined above)

/**
 * @brief Main per-thread dispatch function (single-writer mode)
 * Routes each thread to appropriate function based on pattern configuration
 */
__device__ void dispatch_gpu_thread(
    int bid, int tid,
    bufferElement *buffer,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    // Fetch config from constant memory
    ThreadConfig cfg = d_pattern_gpu[bid][tid];
    
    switch (cfg.role) {
        case ThreadRole::WRITER:
            dispatch_writer(cfg, buffer, r_signal, 
                          w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                          fallback_signal);
            break;
            
        case ThreadRole::READER:
            dispatch_reader(cfg, buffer, results, r_signal,
                          w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                          fallback_signal, timing);
            break;
            
        case ThreadRole::DUMMY_READER:
            gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
            break;
        case ThreadRole::DUMMY_WRITER:
            gpu_dummy_writer_worker_propagation(dummy_buffer, r_signal);
            break;
        case ThreadRole::INACTIVE:
            // Signal readiness so writer doesn't deadlock waiting for inactive threads
            r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
            // Don't time dummy or inactive threads
            {
                int global_tid = bid * GPU_NUM_THREADS + tid;
                timing[global_tid].consumer_type = 0;  // inactive
                timing[global_tid].start_time = 0;
                timing[global_tid].flag_trigger_time = 0;
                timing[global_tid].end_time = 0;
            }
            break;
        default:
            // Do nothing
            break;
    }
}

/**
 * @brief Main per-thread dispatch function (multi-writer mode)
 * Routes each thread to scope-specific buffers
 */
__device__ void dispatch_gpu_thread_multi(
    int bid, int tid,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    // Fetch config from constant memory
    ThreadConfig cfg = d_pattern_gpu[bid][tid];
    
    switch (cfg.role) {
        case ThreadRole::WRITER:
            dispatch_multi_writer(cfg, buffer_t, buffer_b, buffer_d, buffer_s,
                                r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                fallback_signal);
            break;
            
        case ThreadRole::READER:
            dispatch_multi_reader(cfg, buffer_t, buffer_b, buffer_d, buffer_s,
                                results, r_signal,
                                w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                fallback_signal, timing);
            break;
            
        case ThreadRole::DUMMY_READER:
            gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
        case ThreadRole::DUMMY_WRITER:
            gpu_dummy_writer_worker_propagation(dummy_buffer, r_signal);
        case ThreadRole::INACTIVE:
            // Signal readiness so writer doesn't deadlock waiting for inactive threads
            r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
            // Don't time dummy or inactive threads
            {
                int global_tid = bid * GPU_NUM_THREADS + tid;
                timing[global_tid].consumer_type = 0;  // inactive
                timing[global_tid].start_time = 0;
                timing[global_tid].flag_trigger_time = 0;
                timing[global_tid].end_time = 0;
            }
            break;
        default:
            // Do nothing
            break;
    }
}

/**
 * @brief Generic orchestrator kernel (single-writer mode)
 * Launches pattern-based execution for all GPU threads
 */
__global__ void pattern_orchestrator(
    bufferElement *buffer,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    dispatch_gpu_thread(blockIdx.x, threadIdx.x,
                       buffer, dummy_buffer, results,
                       r_signal, w_t_signal, w_b_signal,
                       w_d_signal, w_s_signal, fallback_signal, timing);
}

/**
 * @brief Generic orchestrator kernel (multi-writer mode)
 * Launches pattern-based execution with scope-specific buffers
 */
__global__ void pattern_orchestrator_multi(
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_timing_data *timing
) {
    dispatch_gpu_thread_multi(blockIdx.x, threadIdx.x,
                             buffer_t, buffer_b, buffer_d, buffer_s,
                             dummy_buffer, results,
                             r_signal, w_t_signal, w_b_signal,
                             w_d_signal, w_s_signal, fallback_signal, timing);
}

#endif // PATTERN_DISPATCH_CUH
