#ifndef TYPES_H
#define TYPES_H

#include <cuda/atomic>
#include <stdint.h>

#define BUFFER_SIZE 512
#define NUM_ITERATIONS 10000

#define GPU_NUM_BLOCKS 8
#define GPU_NUM_THREADS 64

#define CPU_NUM_THREADS 32

#define PAGE_SIZE 4096

#define CONSUMERS_CACHE

#ifdef P_H_FLAG_STORE_ORDER_REL
#define P_H_FLAG_STORE_ORDER cuda::memory_order_release
#elif defined(P_H_FLAG_STORE_ORDER_RLX)
#define P_H_FLAG_STORE_ORDER cuda::memory_order_relaxed
#endif

#ifdef C_H_FLAG_LOAD_ORDER_ACQ
#define C_H_FLAG_LOAD_ORDER cuda::memory_order_acquire
#elif defined(C_H_FLAG_LOAD_ORDER_RLX)
#define C_H_FLAG_LOAD_ORDER cuda::memory_order_relaxed
#endif

#ifdef DATA_SIZE_64
typedef uint64_t DATA_SIZE;
#elif defined(DATA_SIZE_32)
typedef uint32_t DATA_SIZE;
#elif defined(DATA_SIZE_16)
typedef uint16_t DATA_SIZE;
#elif defined(DATA_SIZE_8)
typedef uint8_t DATA_SIZE;
#endif

#ifdef CUDA_THREAD_SCOPE_THREAD
#define CUDA_THREAD_SCOPE cuda::thread_scope_thread
#elif defined(CUDA_THREAD_SCOPE_BLOCK)
#define CUDA_THREAD_SCOPE cuda::thread_scope_block
#elif defined(CUDA_THREAD_SCOPE_DEVICE)
#define CUDA_THREAD_SCOPE cuda::thread_scope_device
#elif defined(CUDA_THREAD_SCOPE_SYSTEM)
#define CUDA_THREAD_SCOPE cuda::thread_scope_system
#endif

/**
 * @brief Memory allocator selection
 * 
 * Determines where buffers are allocated in the memory hierarchy.
 * Affects cache coherence behavior and propagation delays.
 */
typedef enum {
    CE_SYS_MALLOC,    ///< System malloc() - standard host memory
    CE_CUDA_MALLOC,   ///< cudaMalloc() - GPU device memory (GPU-only)
    CE_NUMA_HOST,     ///< NUMA node 0 - CPU-local memory
    CE_NUMA_DEVICE,   ///< NUMA node 1 - GPU-local memory
    CE_DRAM,          ///< cudaMallocHost() - pinned host memory
    CE_UM             ///< cudaMallocManaged() - unified memory
} AllocatorType;

/**
 * @brief Consumer type selection
 * 
 * Determines whether GPU or CPU threads act as readers/writers.
 */
typedef enum {
    CE_GPU,           ///< GPU threads as consumer
    CE_CPU            ///< CPU threads as consumer
} ReaderWriterType;

/**
 * @brief Writer spawn control
 * 
 * Controls which device spawns the writer thread in propagation tests.
 */
typedef enum {
    CE_NO_WRITER,     ///< No writer spawned (reader-only mode)
    CE_WRITER,        ///< Single writer (homogeneous, same device as readers)
    CE_HET_WRITER,    ///< Heterogeneous writer (cross-device)
    CE_MULTI_WRITER   ///< Multiple concurrent writers (multi-producer mode)
} WriterType;

/**
 * @brief Buffer element with page-aligned padding
 * 
 * Each element padded to PAGE_SIZE (4KB) to prevent false sharing
 * and isolate cache line effects. Uses system scope atomics by default.
 */
typedef struct bufferElement {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement;

/** @brief Buffer element with thread scope atomics - for multi-writer tests */
typedef struct bufferElement_t {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_t;

/** @brief Buffer element with block scope atomics - for multi-writer tests */
typedef struct bufferElement_b {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_block> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_b;

/** @brief Buffer element with device scope atomics - for multi-writer tests */
typedef struct bufferElement_d {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_device> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_d;

/** @brief Buffer element with system scope atomics - for multi-writer tests */
typedef struct bufferElement_s {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_system> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_s;

/** @brief Non-atomic buffer element - for result storage */
typedef struct bufferElement_na {
    uint32_t data;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} bufferElement_na;

/**
 * @brief Synchronization flags with page-aligned padding
 * 
 * Flags signal readiness between readers and writers.
 * - r_signal: Reader readiness counter (incremented by each reader)
 * - w_signal: Writer completion flag (set to 1 when data ready)
 * - fallback_signal: Timeout mechanism to prevent infinite waits
 * 
 * Scope variants (_t/_b/_d/_s) match corresponding buffer types.
 */

typedef struct flag_t {
    cuda::atomic<uint32_t, cuda::thread_scope_thread> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_t;

typedef struct flag_b {
    cuda::atomic<uint32_t, cuda::thread_scope_block> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_b;

typedef struct flag_d {
    cuda::atomic<uint32_t, cuda::thread_scope_device> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_d;

typedef struct flag_s {
    cuda::atomic<uint32_t, cuda::thread_scope_system> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_s;

/**
 * @brief GPU timing data for performance instrumentation
 * 
 * Records timing information for GPU reader threads.
 * Page-aligned to prevent false sharing across threads.
 * Clock values are from clock64() GPU timer.
 */
typedef struct gpu_timing_data {
    clock_t start_time;         ///< Thread entry time
    clock_t flag_trigger_time;  ///< Time when flag was observed
    clock_t end_time;           ///< Thread exit time
    uint32_t block_id;          ///< Block ID
    uint32_t thread_id;         ///< Thread ID within block
    uint8_t consumer_type;      ///< 0=inactive, 1=reader_acq, 2=reader_rlx, 3=dummy
    uint8_t flag_type;          ///< 0=thread, 1=block, 2=device, 3=system
    bool caching;                 ///< Whether caching variant was used
    char padding[PAGE_SIZE - sizeof(clock_t)*3 - sizeof(uint32_t)*2 - sizeof(uint8_t)*2];
} gpu_timing_data;

/**
 * @brief CPU timing data for performance instrumentation
 * 
 * Records timing information for CPU reader threads.
 * Page-aligned to prevent false sharing across threads.
 * Time values are nanoseconds from std::chrono::high_resolution_clock.
 */
typedef struct cpu_timing_data {
    uint64_t start_ns;          ///< Thread entry time (nanoseconds)
    uint64_t flag_trigger_ns;   ///< Time when flag was observed (nanoseconds)
    uint64_t end_ns;            ///< Thread exit time (nanoseconds)
    uint32_t thread_id;         ///< CPU thread/core ID
    uint8_t consumer_type;      ///< 0=inactive, 1=reader_acq, 2=reader_rlx, 3=dummy
    uint8_t flag_type;          ///< 0=thread, 1=block, 2=device, 3=system
    bool caching;                 ///< Whether caching variant was used
    char padding[PAGE_SIZE - sizeof(uint64_t)*3 - sizeof(uint32_t) - sizeof(uint8_t)*2];
} cpu_timing_data;

#endif // TYPES_H
