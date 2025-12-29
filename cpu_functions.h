#include "types.h"

/**
 * @brief CPU writer for propagation hierarchy testing (homogeneous CPU-only)
 * @param buffer Main data buffer
 * @param r_signal Reader readiness counter (waits for CPU_NUM_THREADS - 1)
 * @param w_t_signal Thread scope writer flag
 * @param w_b_signal Block scope writer flag
 * @param w_d_signal Device scope writer flag
 * @param w_s_signal System scope writer flag
 * @param fallback_signal Timeout mechanism
 * 
 * TIMING POINTS: Use std::chrono::high_resolution_clock for each phase:
 *   Phase 1: Write 10 + set w_t_signal → sleep(5)
 *   Phase 2: Write 20 + set w_b_signal → sleep(5)
 *   Phase 3: Write 30 + set w_d_signal → sleep(5)
 *   Phase 4: Write 40 + set w_s_signal → sleep(5)
 * 
 * This is the MAIN CPU WRITER for single-writer propagation hierarchy tests
 * Uses sleep(5) = 5 seconds between phases
 */
static void __attribute__((optimize("O0"))) cpu_buffer_writer_propagation_hierarchy(bufferElement *buffer, flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal) {
    
    printf("CPU Writer %d\n", sched_getcpu());

    // int core_id = sched_getcpu();
    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 1) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    
    // Set Writer Signals (Thread)
    w_t_signal->flag.store(1, cuda::memory_order_release);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Block)
    w_b_signal->flag.store(1, cuda::memory_order_release);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (Device)
    w_d_signal->flag.store(1, cuda::memory_order_release);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    
    // Set Writer Signals (System)
    w_s_signal->flag.store(1, cuda::memory_order_release);
    
    sleep(5);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    sleep(5);

    fallback_signal->flag.store(4, cuda::memory_order_release);

}

static void __attribute__((optimize("O0"))) cpu_buffer_writer_propagation_hierarchy_gpu(bufferElement *buffer, flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal) {

    printf("CPU Het-Writer %d %d\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 1);
    // int core_id = sched_getcpu();

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

    w_t_signal->flag.store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_b_signal->flag.store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_d_signal->flag.store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_s_signal->flag.store(1, cuda::memory_order_relaxed);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(50, cuda::memory_order_relaxed);
    }

    sleep(5);

    fallback_signal->flag.store(4, cuda::memory_order_release);
}

template <typename B, typename R, typename W, typename F>
static void __attribute__((optimize("O0"))) cpu_buffer_reader_propagation_hierarchy(B *buffer, bufferElement_na * results, R *r_signal, W *w_signal, F *fallback_signal) {
    
    int core_id = sched_getcpu();

    uint init_flag = w_signal->flag.load(cuda::memory_order_relaxed);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id].data = result;

    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while (w_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0 && fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id % CPU_NUM_THREADS].data = result;

    printf("C[%d:%d:%d] Result %d\n", core_id, core_id % 8, core_id % 4, results[core_id % CPU_NUM_THREADS].data);
}

template <typename B, typename R, typename W, typename F>
static void __attribute__((optimize("O0"))) cpu_buffer_multi_reader_propagation_hierarchy(B *buffer, bufferElement_na * results, R *r_signal, flag_t * w_t_signal, flag_b * w_b_signal, flag_d * w_d_signal, flag_s * w_s_signal, F * fallback_signal) {
    
    int core_id = sched_getcpu();

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

    results[core_id].data = result;

    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);

    result = 0;

    while ((init_flag_t = w_t_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_b = w_b_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_d = w_d_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && (init_flag_s = w_s_signal->flag.load(C_H_FLAG_LOAD_ORDER) == 0) && fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait for Writer Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    results[core_id % CPU_NUM_THREADS].data = result + init_flag_t * 1000000000 + init_flag_b * 100000000 + init_flag_d * 10000000 + init_flag_s * 1000000;

    printf("C[%d:%d:%d] Result %d\n", core_id, core_id % 8, core_id % 4, results[core_id % CPU_NUM_THREADS].data);
}




template <typename R>
static void __attribute__((optimize("O0"))) cpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, R *r_signal) {
    int core_id = sched_getcpu();
    
    uint result = 0;
    
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < BUFFER_SIZE; j++) {
            result += buffer[j].data.load(cuda::memory_order_relaxed);
        }
    }
    
    results[core_id % CPU_NUM_THREADS].data = result;
}

/**
 * @brief CPU orchestrator function for propagation hierarchy testing
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
 * Thread role assignment based on core_id % 8:
 *   Core 0: Writer (if spawn_writer != CE_NO_WRITER)
 *   Cores 0-3 (mod 8): Relaxed readers (thread/block/device/system)
 *   Cores 4-7 (mod 8): Acquire readers (or dummy if NO_ACQ)
 * 
 * NO TIMING NEEDED - Individual consumer functions handle their own timing
 */
static void __attribute__((optimize("O0"))) cpu_buffer_reader_writer_propagation_hierarchy(bufferElement *buffer, bufferElement *w_buffer, bufferElement_na * results, flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal, WriterType *spawn_writer) {

    
    int core_id = sched_getcpu();
    
    if ((*spawn_writer == CE_WRITER || *spawn_writer == CE_HET_WRITER) && core_id % 32 == 0) {
        // cpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        if (*spawn_writer == WriterType::CE_WRITER) {
            cpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            cpu_buffer_writer_propagation_hierarchy_gpu(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else {
        if (core_id % 8 == 0) {
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_t_signal, fallback_signal);
        } else if (core_id % 8 == 1) {
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_b_signal, fallback_signal);
        } else if (core_id % 8 == 2) {
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_d_signal, fallback_signal);
        } else if (core_id % 8 == 3) {
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_s_signal, fallback_signal);
        } else if (core_id % 8 == 4) {
            #ifdef NO_ACQ
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            #else
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_t_signal, fallback_signal);
            #endif
        } else if (core_id % 8 == 5) {
            #ifdef NO_ACQ
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            #else
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_b_signal, fallback_signal);
            #endif
        } else if (core_id % 8 == 6) {
            #ifdef NO_ACQ
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            #else
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_d_signal, fallback_signal);
            #endif
        } else if (core_id % 8 == 7) {
            #ifdef NO_ACQ
            cpu_dummy_reader_worker_propagation(w_buffer, results, r_signal);
            #else
            cpu_buffer_reader_propagation_hierarchy(buffer, results, r_signal, w_s_signal, fallback_signal);
            #endif
        }
    }
}

/**
 * @brief CPU multi-writer for THREAD scope (homogeneous CPU-only)
 * @param buffer Thread-scoped atomic buffer
 * @param r_signal Reader readiness counter (waits for CPU_NUM_THREADS - 4)
 * @param w_signal Thread scope writer flag
 * @param fb_signal Fallback timeout mechanism
 * 
 * TIMING POINT: Use std::chrono - time write loop (value 10) + flag store → sleep(5) → write (value 1)
 * Part of multi-writer pattern with 4 concurrent CPU writers
 */
static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_thread_propagation_hierarchy(bufferElement_t * buffer, flag_d * r_signal, flag_t * w_signal, flag_s * fb_signal) {
    
    printf("CPU Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS - 4);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_block_propagation_hierarchy(bufferElement_b * buffer, flag_d * r_signal, flag_b * w_signal, flag_s * fb_signal) {
    
    printf("CPU Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS - 4);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_device_propagation_hierarchy(bufferElement_d * buffer, flag_d * r_signal, flag_d * w_signal, flag_s * fb_signal) {
    
    printf("CPU Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS - 4);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_system_propagation_hierarchy(bufferElement_s * buffer, flag_d * r_signal, flag_s * w_signal, flag_s * fb_signal) {
    
    printf("CPU Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS - 4);

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_thread_propagation_hierarchy_gpu(bufferElement_t * buffer, flag_d * r_signal, flag_t * w_signal, flag_s * fb_signal) {
    
    printf("CPU Het-Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4);
    // int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(1, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_block_propagation_hierarchy_gpu(bufferElement_b * buffer, flag_d * r_signal, flag_b * w_signal, flag_s * fb_signal) {
    
    printf("CPU Het-Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4);
    // int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(2, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_device_propagation_hierarchy_gpu(bufferElement_d * buffer, flag_d * r_signal, flag_d * w_signal, flag_s * fb_signal) {
    
    printf("CPU Het-Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4);
    // int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(3, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_multi_writer_system_propagation_hierarchy_gpu(bufferElement_s * buffer, flag_d * r_signal, flag_s * w_signal, flag_s * fb_signal) {
    
    printf("CPU Het-Writer (Core: %d) (Waiting For: %d)\n", sched_getcpu(), CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4);
    // int core_id = sched_getcpu();

    uint result = 0;

    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }

    // sleep(5);

    while (r_signal->flag.load(cuda::memory_order_relaxed) != CPU_NUM_THREADS + (GPU_NUM_BLOCKS * GPU_NUM_THREADS) - 4) {
        // Wait for Reader Signal
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }

    w_signal->flag.store(1, P_H_FLAG_STORE_ORDER);

    sleep(5);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(4, cuda::memory_order_relaxed);
    }

    fb_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
}

static void __attribute__((optimize("O0"))) cpu_buffer_reader_multi_writer_propagation_hierarchy(bufferElement * dummy_buffer, bufferElement_t * buffer_t, bufferElement_b * buffer_b, bufferElement_d * buffer_d, bufferElement_s * buffer_s, bufferElement_na * results, flag_d * r_signal, flag_t * w_signal_t, flag_b * w_signal_b, flag_d * w_signal_d, flag_s * w_signal_s, flag_s * fb_signal, WriterType * spawn_writer) {
    
    int core_id = sched_getcpu();
    
    if (*spawn_writer != CE_NO_WRITER && core_id % 8 == 0) {
        switch (core_id / 8) {
            case 0:
                if (*spawn_writer == CE_HET_WRITER) {
                    cpu_buffer_multi_writer_thread_propagation_hierarchy_gpu(buffer_t, r_signal, w_signal_t, fb_signal);
                } else {
                    cpu_buffer_multi_writer_thread_propagation_hierarchy(buffer_t, r_signal, w_signal_t, fb_signal);
                }
                break;
            case 1:
                if (*spawn_writer == CE_HET_WRITER) {
                    cpu_buffer_multi_writer_block_propagation_hierarchy_gpu(buffer_b, r_signal, w_signal_b, fb_signal);
                } else {
                    cpu_buffer_multi_writer_block_propagation_hierarchy(buffer_b, r_signal, w_signal_b, fb_signal);
                }
                break;
            case 2:
                if (*spawn_writer == CE_HET_WRITER) {
                    cpu_buffer_multi_writer_device_propagation_hierarchy_gpu(buffer_d, r_signal, w_signal_d, fb_signal);
                } else {
                    cpu_buffer_multi_writer_device_propagation_hierarchy(buffer_d, r_signal, w_signal_d, fb_signal);
                }
                break;
            case 3:
                if (*spawn_writer == CE_HET_WRITER) {
                    cpu_buffer_multi_writer_system_propagation_hierarchy_gpu(buffer_s, r_signal, w_signal_s, fb_signal);
                } else {
                    cpu_buffer_multi_writer_system_propagation_hierarchy(buffer_s, r_signal, w_signal_s, fb_signal);
                }
                break;
            default:
                break;
        }
    } else {
        switch (core_id % 8) {
            case 0:
                cpu_buffer_reader_propagation_hierarchy(buffer_t, results, r_signal, w_signal_t, fb_signal);
                break;
            case 1:
                cpu_buffer_reader_propagation_hierarchy(buffer_b, results, r_signal, w_signal_b, fb_signal);
                break;
            case 2:
                cpu_buffer_reader_propagation_hierarchy(buffer_d, results, r_signal, w_signal_d, fb_signal);
                break;
            case 3:
                cpu_buffer_reader_propagation_hierarchy(buffer_s, results, r_signal, w_signal_s, fb_signal);
                break;
            case 4:
                cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 5:
                cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 6:
                cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            case 7:
                cpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
                break;
            default:
                break;
        }
    }
}