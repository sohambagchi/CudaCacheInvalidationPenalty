#ifndef PATTERN_DISPATCH_CPU_H
#define PATTERN_DISPATCH_CPU_H

#include "pattern_config.hpp"
#include "types.hpp"
#include <cuda/atomic>
#include <sched.h>

// ============================================================================
// CPU READER VARIANTS
// ============================================================================

/**
 * @brief CPU buffer reader with ACQUIRE memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_acquire(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal
) {
    int core_id = sched_getcpu();
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
    printf("C[%d] ACQ Result %d\n", core_id, result);
}

/**
 * @brief CPU buffer reader with RELAXED memory ordering
 */
template <typename B, typename W, typename R>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_relaxed(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal
) {
    int core_id = sched_getcpu();
    uint result = 0;
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
    printf("C[%d] RLX Result %d\n", core_id, result);
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
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 1) {}
    
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
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 1) {}
    
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
    flag_s *fallback_signal
) {
    if (cfg.ordering == MemoryOrdering::ACQUIRE) {
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                cpu_buffer_reader_acquire(buffer_t, results, r_signal, w_t_signal, fallback_signal);
                break;
            case ThreadScope::BLOCK:
                cpu_buffer_reader_acquire(buffer_b, results, r_signal, w_b_signal, fallback_signal);
                break;
            case ThreadScope::DEVICE:
                cpu_buffer_reader_acquire(buffer_d, results, r_signal, w_d_signal, fallback_signal);
                break;
            case ThreadScope::SYSTEM:
                cpu_buffer_reader_acquire(buffer_s, results, r_signal, w_s_signal, fallback_signal);
                break;
        }
    } else {  // RELAXED
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                cpu_buffer_reader_relaxed(buffer_t, results, r_signal, w_t_signal, fallback_signal);
                break;
            case ThreadScope::BLOCK:
                cpu_buffer_reader_relaxed(buffer_b, results, r_signal, w_b_signal, fallback_signal);
                break;
            case ThreadScope::DEVICE:
                cpu_buffer_reader_relaxed(buffer_d, results, r_signal, w_d_signal, fallback_signal);
                break;
            case ThreadScope::SYSTEM:
                cpu_buffer_reader_relaxed(buffer_s, results, r_signal, w_s_signal, fallback_signal);
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
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    if (!g_active_pattern) return;
    
    ThreadConfig cfg = g_active_pattern->cpu_threads[tid];
    
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
                    cpu_buffer_reader_acquire(buffer, results, r_signal,
                                            w_t_signal, fallback_signal);
                    break;
                case ThreadScope::BLOCK:
                    cpu_buffer_reader_acquire(buffer, results, r_signal,
                                            w_b_signal, fallback_signal);
                    break;
                case ThreadScope::DEVICE:
                    cpu_buffer_reader_acquire(buffer, results, r_signal,
                                            w_d_signal, fallback_signal);
                    break;
                case ThreadScope::SYSTEM:
                    cpu_buffer_reader_acquire(buffer, results, r_signal,
                                            w_s_signal, fallback_signal);
                    break;
            }
        } else {  // RELAXED
            switch (cfg.watch_flag) {
                case ThreadScope::THREAD:
                    cpu_buffer_reader_relaxed(buffer, results, r_signal,
                                            w_t_signal, fallback_signal);
                    break;
                case ThreadScope::BLOCK:
                    cpu_buffer_reader_relaxed(buffer, results, r_signal,
                                            w_b_signal, fallback_signal);
                    break;
                case ThreadScope::DEVICE:
                    cpu_buffer_reader_relaxed(buffer, results, r_signal,
                                            w_d_signal, fallback_signal);
                    break;
                case ThreadScope::SYSTEM:
                    cpu_buffer_reader_relaxed(buffer, results, r_signal,
                                            w_s_signal, fallback_signal);
                    break;
            }
        }
    }
    // Note: DUMMY_READER and DUMMY_WRITER would need implementation if used on CPU
}

/**
 * @brief Dispatch CPU thread to appropriate function based on pattern (multi-writer mode)
 */
static void dispatch_cpu_thread_multi(
    int tid,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
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
                                fallback_signal);
    }
    // Note: DUMMY_READER and DUMMY_WRITER would need implementation if used on CPU
}

#endif // PATTERN_DISPATCH_CPU_H
