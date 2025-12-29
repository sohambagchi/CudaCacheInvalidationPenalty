#ifndef PATTERN_DISPATCH_CPU_H
#define PATTERN_DISPATCH_CPU_H

#include "pattern_config.hpp"
#include "types.hpp"
#include <cuda/atomic>
#include <sched.h>
#include <chrono>

// ============================================================================
// CPU READER VARIANTS
// ============================================================================

/**
 * @brief CPU buffer reader with ACQUIRE memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_acquire_no_cache(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
    uint result = 0;
    
    timing[tid].thread_id = tid;
    timing[tid].consumer_type = 1;  // reader_acq
    timing[tid].caching = false;
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    timing[tid].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    
    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    auto flag_time = std::chrono::high_resolution_clock::now();
    timing[tid].flag_trigger_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(flag_time.time_since_epoch()).count();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    timing[tid].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
    
    printf("C[%d] NC-ACQ Result %d\n", core_id, result);
}

/**
 * @brief CPU buffer reader with ACQUIRE memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_acquire_caching(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
    uint result = 0;
    
    timing[tid].thread_id = tid;
    timing[tid].consumer_type = 1;  // reader_acq
    timing[tid].caching = true;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    timing[tid].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    
    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    auto flag_time = std::chrono::high_resolution_clock::now();
    timing[tid].flag_trigger_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(flag_time.time_since_epoch()).count();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    timing[tid].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
    
    printf("C[%d] C-ACQ Result %d\n", core_id, result);
}

/**
 * @brief CPU buffer reader with RELAXED memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_relaxed_no_cache(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
    uint result = 0;
    
    timing[tid].thread_id = tid;
    timing[tid].consumer_type = 2;  // reader_rlx
    timing[tid].caching = false;
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    timing[tid].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    
    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    auto flag_time = std::chrono::high_resolution_clock::now();
    timing[tid].flag_trigger_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(flag_time.time_since_epoch()).count();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    timing[tid].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
    
    printf("C[%d] NC-RLX Result %d\n", core_id, result);
}
/**
 * @brief CPU buffer reader with RELAXED memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_relaxed_caching(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    timing[tid].thread_id = tid;
    timing[tid].consumer_type = 2;  // reader_rlx
    timing[tid].caching = true;
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    timing[tid].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();

    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    // Record flag trigger time
    auto flag_time = std::chrono::high_resolution_clock::now();
    timing[tid].flag_trigger_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(flag_time.time_since_epoch()).count();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    timing[tid].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
    
    printf("C[%d] C-RLX Result %d\n", core_id, result);
}

// ============================================================================
// CPU WRITER VARIANTS
// ============================================================================

/**
 * @brief CPU buffer writer with RELEASE memory ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_writer_release(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("CPU Writer (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fallback_signal->flag.store(4, cuda::memory_order_release);
}

/**
 * @brief CPU buffer writer with RELAXED memory ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_writer_relaxed(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("CPU Writer (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fallback_signal->flag.store(4, cuda::memory_order_release);
}

// ============================================================================
// CPU DUMMY FUNCTIONS FOR BACKGROUND LOAD
// ============================================================================

/**
 * @brief Dummy CPU reader for background load
 * Performs reads without synchronization to create memory traffic
 */
static void __attribute__((optimize("O0"))) cpu_dummy_reader_worker_propagation(
    bufferElement *dummy_buffer,
    bufferElement_na *results, 
    flag_d *r_signal
) {
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
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
 * @brief Dummy CPU writer for background load
 * Performs writes without synchronization to create memory traffic
 */
static void __attribute__((optimize("O0"))) cpu_dummy_writer_worker_propagation(
    bufferElement *dummy_buffer, 
    flag_d *r_signal
) {
    // Wait for readers to be ready (same logic as GPU)
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    // Background write load
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            dummy_buffer[i].data.store(iter + i, cuda::memory_order_relaxed);
        }
    }
}

// ============================================================================
// CPU MULTI-WRITER VARIANTS
// ============================================================================

/**
 * @brief Multi-writer thread-scope with RELEASE ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_thread_release(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer THREAD (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer thread-scope with RELAXED ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_thread_relaxed(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer THREAD (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer block-scope with RELEASE ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_block_release(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer BLOCK (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer block-scope with RELAXED ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_block_relaxed(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer BLOCK (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer device-scope with RELEASE ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_device_release(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer DEVICE (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer device-scope with RELAXED ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_device_relaxed(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer DEVICE (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer system-scope with RELEASE ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_system_release(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer SYSTEM (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_release);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

/**
 * @brief Multi-writer system-scope with RELAXED ordering
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_system_relaxed(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
) {
    printf("CPU Multi-Writer SYSTEM (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS + CPU_NUM_THREADS - 4) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_signal->flag.store(1, cuda::memory_order_relaxed);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }
    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

// ============================================================================
// CPU DISPATCH LOGIC
// ============================================================================

// Store the active pattern for CPU threads
extern const PatternConfig* g_active_pattern;

/**
 * @brief Helper to convert ThreadScope to uint8_t for timing data
 */
static inline uint8_t scope_to_uint8_cpu(ThreadScope scope) {
    switch (scope) {
        case ThreadScope::THREAD: return 0;
        case ThreadScope::BLOCK: return 1;
        case ThreadScope::DEVICE: return 2;
        case ThreadScope::SYSTEM: return 3;
        default: return 2;  // Default to device
    }
}

/**
 * @brief Dispatch multi-writer to appropriate scope-specific variant
 */
static void dispatch_multi_writer_cpu(
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
                cpu_buffer_multi_writer_thread_release(buffer_t, r_signal, w_t_signal, fallback_signal);
            else
                cpu_buffer_multi_writer_thread_relaxed(buffer_t, r_signal, w_t_signal, fallback_signal);
            break;
        case ThreadScope::BLOCK:
            if (is_release)
                cpu_buffer_multi_writer_block_release(buffer_b, r_signal, w_b_signal, fallback_signal);
            else
                cpu_buffer_multi_writer_block_relaxed(buffer_b, r_signal, w_b_signal, fallback_signal);
            break;
        case ThreadScope::DEVICE:
            if (is_release)
                cpu_buffer_multi_writer_device_release(buffer_d, r_signal, w_d_signal, fallback_signal);
            else
                cpu_buffer_multi_writer_device_relaxed(buffer_d, r_signal, w_d_signal, fallback_signal);
            break;
        case ThreadScope::SYSTEM:
            if (is_release)
                cpu_buffer_multi_writer_system_release(buffer_s, r_signal, w_s_signal, fallback_signal);
            else
                cpu_buffer_multi_writer_system_relaxed(buffer_s, r_signal, w_s_signal, fallback_signal);
            break;
    }
}

/**
 * @brief Dispatch multi-reader to scope-matching buffer
 */
template <typename B_T, typename B_B, typename B_D, typename B_S>
static void dispatch_multi_reader_cpu(
    ThreadConfig cfg,
    B_T *buffer_t, B_B *buffer_b,
    B_D *buffer_d, B_S *buffer_s,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    // Set flag type in timing data
    int core_id = sched_getcpu();
    int tid = core_id % CPU_NUM_THREADS;
    timing[tid].flag_type = scope_to_uint8_cpu(cfg.watch_flag);
    
    if (cfg.ordering == MemoryOrdering::ACQUIRE) {
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching) {
                    cpu_buffer_reader_acquire_caching(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_acquire_no_cache(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                }
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching) {
                    cpu_buffer_reader_acquire_caching(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_acquire_no_cache(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                }    
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching) {
                    cpu_buffer_reader_acquire_caching(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_acquire_no_cache(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                }
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching) {
                    cpu_buffer_reader_acquire_caching(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_acquire_no_cache(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                }
                break;
        }
    } else {  // RELAXED
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                if (cfg.caching) {
                    cpu_buffer_reader_relaxed_caching(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_relaxed_no_cache(buffer_t, results, r_signal, w_t_signal, fallback_signal, timing);
                }
                break;
            case ThreadScope::BLOCK:
                if (cfg.caching) {
                    cpu_buffer_reader_relaxed_caching(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_relaxed_no_cache(buffer_b, results, r_signal, w_b_signal, fallback_signal, timing);
                }
                break;
            case ThreadScope::DEVICE:
                if (cfg.caching) {
                    cpu_buffer_reader_relaxed_caching(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_relaxed_no_cache(buffer_d, results, r_signal, w_d_signal, fallback_signal, timing);
                }
                break;
            case ThreadScope::SYSTEM:
                if (cfg.caching) {
                    cpu_buffer_reader_relaxed_caching(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                } else {
                    cpu_buffer_reader_relaxed_no_cache(buffer_s, results, r_signal, w_s_signal, fallback_signal, timing);
                }
                break;
        }
    }
}

/**
 * @brief Dispatch CPU thread to appropriate function based on pattern (single-writer mode)
 */
static void dispatch_cpu_thread(
    int tid,
    bufferElement *buffer,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    if (!g_active_pattern) return;
    
    ThreadConfig cfg = g_active_pattern->cpu_threads[tid];
    
    // Set flag type for readers
    timing[tid].flag_type = scope_to_uint8_cpu(cfg.watch_flag);
    
    if (cfg.role == ThreadRole::WRITER) {
        if (cfg.ordering == MemoryOrdering::RELEASE) {
            cpu_buffer_writer_release(buffer, r_signal,
                                     w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                     fallback_signal);
        } else {
            cpu_buffer_writer_relaxed(buffer, r_signal,
                                     w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                     fallback_signal);
        }
    }
    else if (cfg.role == ThreadRole::READER) {
        if (cfg.ordering == MemoryOrdering::ACQUIRE) {
            switch (cfg.watch_flag) {
                case ThreadScope::THREAD:
                    if (cfg.caching) {
                        cpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                                w_t_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                                w_t_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::BLOCK:
                    if (cfg.caching) {
                        cpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                                w_b_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                                w_b_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::DEVICE:
                    if (cfg.caching) {
                        cpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                                w_d_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                                w_d_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::SYSTEM:
                    if (cfg.caching) {
                        cpu_buffer_reader_acquire_caching(buffer, results, r_signal,
                                                w_s_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_acquire_no_cache(buffer, results, r_signal,
                                                w_s_signal, fallback_signal, timing);
                    }
                    break;
            }
        } else {  // RELAXED
            switch (cfg.watch_flag) {
                case ThreadScope::THREAD:
                    if (cfg.caching) {
                        cpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                                w_t_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                                w_t_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::BLOCK:
                    if (cfg.caching) {
                        cpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                                w_b_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                                w_b_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::DEVICE:
                    if (cfg.caching) {
                        cpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                                w_d_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                                w_d_signal, fallback_signal, timing);
                    }
                    break;
                case ThreadScope::SYSTEM:
                    if (cfg.caching) {
                        cpu_buffer_reader_relaxed_caching(buffer, results, r_signal,
                                                w_s_signal, fallback_signal, timing);
                    } else {
                        cpu_buffer_reader_relaxed_no_cache(buffer, results, r_signal,
                                                w_s_signal, fallback_signal, timing);
                    }
                    break;
            }
        }
    } else if (cfg.role == ThreadRole::INACTIVE) {
        // Mark as inactive
        r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
        timing[tid].consumer_type = 0;
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
    } else if (cfg.role == ThreadRole::DUMMY_READER) {
        // Mark as dummy
        timing[tid].consumer_type = 3;  // dummy
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
        cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
    } else if (cfg.role == ThreadRole::DUMMY_WRITER) {
        // Mark as dummy
        timing[tid].consumer_type = 3;  // dummy
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
        cpu_dummy_writer_worker_propagation(dummy_buffer, r_signal);
    }
}

/**
 * @brief Dispatch CPU thread to appropriate function based on pattern (multi-writer mode)
 */
static void dispatch_cpu_thread_multi(
    int tid,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement * dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    cpu_timing_data *timing
) {
    if (!g_active_pattern) return;
    
    ThreadConfig cfg = g_active_pattern->cpu_threads[tid];
    
    if (cfg.role == ThreadRole::WRITER) {
        dispatch_multi_writer_cpu(cfg, buffer_t, buffer_b, buffer_d, buffer_s,
                                 r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                 fallback_signal);
    }
    else if (cfg.role == ThreadRole::READER) {
        dispatch_multi_reader_cpu(cfg, buffer_t, buffer_b, buffer_d, buffer_s,
                                results, r_signal,
                                w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                fallback_signal, timing);
    } else if (cfg.role == ThreadRole::INACTIVE) {
        // Mark as inactive
        r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
        timing[tid].consumer_type = 0;
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
    } else if (cfg.role == ThreadRole::DUMMY_READER) {
        // Mark as dummy
        timing[tid].consumer_type = 3;  // dummy
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
        // Note: Using buffer_d for dummy operations in multi-writer mode
        cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
    } else if (cfg.role == ThreadRole::DUMMY_WRITER) {
        // Mark as dummy
        timing[tid].consumer_type = 3;  // dummy
        timing[tid].start_ns = 0;
        timing[tid].flag_trigger_ns = 0;
        timing[tid].end_ns = 0;
        // Note: Using buffer_d for dummy operations in multi-writer mode
        cpu_dummy_writer_worker_propagation(dummy_buffer, r_signal);
    }
}

#endif // PATTERN_DISPATCH_CPU_H
