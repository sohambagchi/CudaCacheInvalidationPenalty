# Technical Reference

Complete technical reference for the CUDA Cache Invalidation Penalty Testing framework. This document details all data structures, consumer functions, and timing instrumentation points.

## Table of Contents

1. [Configuration & Constants](#configuration--constants)
2. [Data Structures](#data-structures)
3. [Consumer Functions](#consumer-functions)
4. [Utility Functions](#utility-functions)
5. [Main Execution Flow](#main-execution-flow)
6. [Build System](#build-system)
7. [Function Status Reference](#function-status-quick-reference)

---

## Configuration & Constants

### Compile-Time Flags

#### Thread Scope Selection (mutually exclusive)
- `CUDA_THREAD_SCOPE_THREAD` → `cuda::thread_scope_thread`
- `CUDA_THREAD_SCOPE_BLOCK` → `cuda::thread_scope_block`
- `CUDA_THREAD_SCOPE_DEVICE` → `cuda::thread_scope_device`
- `CUDA_THREAD_SCOPE_SYSTEM` → `cuda::thread_scope_system`

#### Data Size Selection (mutually exclusive)
- `DATA_SIZE_8` → `uint8_t`
- `DATA_SIZE_16` → `uint16_t`
- `DATA_SIZE_32` → `uint32_t`
- `DATA_SIZE_64` → `uint64_t`

#### Memory Ordering for Writer Flags
- `P_H_FLAG_STORE_ORDER_REL` → Producers use `memory_order_release`
- `P_H_FLAG_STORE_ORDER_RLX` → Producers use `memory_order_relaxed`

#### Consumer Behavior
- `NO_ACQ` - Disables acquire loads, readers use dummy operations instead
- `CONSUMERS_CACHE` - Enables initial buffer read before synchronization (pre-caching)

### Runtime Constants

```c
#define BUFFER_SIZE 512          // Number of buffer elements
#define NUM_ITERATIONS 10000     // Iterations per consumer
#define GPU_NUM_BLOCKS 8         // GPU block count
#define GPU_NUM_THREADS 64       // Threads per block (total: 512)
#define CPU_NUM_THREADS 32       // CPU thread count
#define PAGE_SIZE 4096           // Padding size for cache line isolation
```

---

## Data Structures

### Buffer Elements

Each buffer element contains an atomic variable padded to page size (4KB) for cache line isolation.

**Base Structure:**
```c
typedef struct bufferElement {
    cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement;
```

**Scope Variants:**
- `bufferElement_t` - Thread scope atomics
- `bufferElement_b` - Block scope atomics
- `bufferElement_d` - Device scope atomics
- `bufferElement_s` - System scope atomics
- `bufferElement_na` - Non-atomic (for results)

**Timing Consideration:** Each consumer function iterates over `BUFFER_SIZE` elements. Time the entire loop, not individual loads/stores.

### Synchronization Flags

Flags signal readiness between readers and writers.

**Base Structure:**
```c
typedef struct flag_s {
    cuda::atomic<uint32_t, cuda::thread_scope_system> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_s;
```

**Scope Variants:**
- `flag_t` - Thread scope
- `flag_b` - Block scope
- `flag_d` - Device scope
- `flag_s` - System scope

**Usage Pattern:**
- `r_signal` - Reader readiness counter (incremented by each reader)
- `w_signal_*` - Writer completion flags (set to 1 when data ready)
- `fallback_signal` - Timeout mechanism to prevent infinite waits

### Enumerations

```c
typedef enum {
    CE_SYS_MALLOC,    // malloc()
    CE_CUDA_MALLOC,   // cudaMalloc()
    CE_NUMA_HOST,     // numa_alloc_onnode(0)
    CE_NUMA_DEVICE,   // numa_alloc_onnode(1)
    CE_DRAM,          // cudaMallocHost()
    CE_UM             // cudaMallocManaged()
} AllocatorType;

typedef enum {
    CE_GPU,           // GPU consumer
    CE_CPU            // CPU consumer
} ReaderWriterType;

typedef enum {
    CE_NO_WRITER,     // No writer spawned (reader-only)
    CE_WRITER,        // Single writer (homogeneous)
    CE_HET_WRITER,    // Heterogeneous writer (cross-device)
    CE_MULTI_WRITER   // Multiple concurrent writers
} WriterType;
```

---

## Consumer Functions

Consumer functions are the core measurement points. **All active consumer functions should be instrumented with timing.**

### Organization

Functions are organized into three execution patterns:

1. **Simple Reader/Writer** - Legacy/testing functions (mostly unused)
2. **Propagation Hierarchy (Single Writer)** - One writer, multiple readers
3. **Propagation Hierarchy (Multi-Writer)** - Four concurrent writers at different scope levels

### Pattern 1: Simple Reader/Writer

**Status:** Mostly legacy code for initial testing. Not used in main execution path.

#### GPU Functions

**⚠️ REDUNDANT - These are superseded by propagation hierarchy functions**

```c
__global__ void gpu_buffer_reader_single_iter(bufferElement *buffer, uint32_t *results, clock_t *duration)
```
- Single-iteration reader with built-in timing (uses `clock64()`)

```c
__global__ void gpu_buffer_reader(bufferElement *buffer, uint32_t *results, uint32_t *duration)
```
- Multi-iteration reader with per-iteration timing

```c
__device__ void gpu_buffer_reader_diverge(bufferElement *buffer, uint32_t *results, uint32_t *duration)
```
- **NAMING ISSUE:** Name suggests thread divergence testing, but function is simple sequential reader

```c
__device__ void gpu_buffer_reader_diverge_constant(bufferElement *buffer, uint32_t *result)
```
- No timing, returns single aggregated result

```c
__global__ void gpu_buffer_writer_single_iter(bufferElement *buffer, int chunkSize)
```
- Single write pass across buffer with chunking

```c
__global__ void gpu_buffer_writer(bufferElement *buffer, int chunkSize, clock_t *sleep_duration)
```
- Multi-iteration writer with sleep delays

```c
__device__ void gpu_buffer_writer_diverge(bufferElement *buffer, clock_t *sleep_duration)
```
- **NAMING ISSUE:** Name suggests divergence, but is straightforward writer

```c
__global__ void gpu_buffer_reader_writer(bufferElement *buffer, bufferElement *w_buffer, ...)
```
- Combined reader-writer with thread-based role assignment

#### Active Simple Workers (Used as Background Load)

```c
__device__ void gpu_dummy_writer_worker(bufferElement *buffer)
```
- Simple writer without synchronization (keeps GPU busy)
- Used in propagation tests as background load
- **NO TIMING NEEDED**

```c
__device__ void gpu_dummy_reader_worker(bufferElement *buffer, bufferElement_na *results)
```
- Simple reader without synchronization
- Used in propagation tests as background load
- **NO TIMING NEEDED**

#### CPU Functions (All Redundant)

```c
void cpu_buffer_writer_single_iter(bufferElement *buffer)
void cpu_buffer_writer(bufferElement *buffer, struct timespec *sleep_duration)
void cpu_buffer_reader_single_iter(bufferElement *buffer)
void cpu_buffer_reader(bufferElement *buffer, uint32_t *result, std::chrono::duration<uint32_t, std::nano> *duration)
void buffer_reader(bufferElement *buffer)
```

**⚠️ All superseded by propagation hierarchy functions**

---

### Pattern 2: Propagation Hierarchy (Single Writer)

**This is the primary test pattern.** One designated writer updates data at specific intervals (progressively setting flags for each thread scope level), while multiple readers test visibility with different acquire semantics.

#### Synchronization Flow

1. All readers signal ready (`r_signal.fetch_add`)
2. Writer waits for all readers (`r_signal == expected_readers`)
3. Writer updates buffer and sets first flag (thread scope)
4. Writer sleeps (`cudaSleep` or `sleep(5)`)
5. Repeat steps 3-4 for block, device, and system scope flags
6. Readers wait on flags and measure when data becomes visible

#### GPU Reader Functions (Templated)

**⏱️ TIMING INSERTION POINT:** Time from flag wait until buffer read completion

```c
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_propagation_hierarchy_acq(
    B *buffer,                    // Buffer to read (scope-specific type)
    bufferElement_na *results,    // Per-thread result storage
    R *r_signal,                  // Reader readiness signal
    W *w_signal,                  // Writer completion signal (scope-specific)
    flag_s *fallback_signal       // Timeout fallback
)
```
- Uses `memory_order_acquire` on flag load
- **TIMING:** Start when `w_signal` becomes non-zero, end after buffer read loop
- Stores sum of buffer values in `results->data`

```c
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_propagation_hierarchy_rlx(
    B *buffer, bufferElement_na *results, R *r_signal, W *w_signal, flag_s *fallback_signal
)
```
- Uses `memory_order_relaxed` on flag load
- **TIMING:** Same timing points as `_acq` variant for comparison

```c
template <typename B, typename W, typename R>
__device__ void gpu_buffer_multi_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results, R *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)
```
- Waits on ALL four scope-level flags simultaneously
- **TIMING:** Start when ANY flag becomes non-zero, track which flag fired first
- Encodes which flag succeeded in result value (offset by powers of 10)

```c
template <typename B, typename W, typename R>
__device__ void gpu_buffer_multi_reader_propagation_hierarchy_rlx(...)
```
- Relaxed variant of multi-reader

#### GPU Writer Functions

**⏱️ TIMING INSERTION POINT:** Time each write phase (buffer write + flag store)

```c
__device__ void gpu_buffer_writer_propagation_hierarchy(
    bufferElement *buffer,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)
```
- **Homogeneous writer** (GPU-only readers)
- Waits for `GPU_NUM_THREADS * GPU_NUM_BLOCKS - 1` readers
- Writes values 10, 20, 30, 40 sequentially
- **TIMING POINTS:**
  - Phase 1: Write loop + thread flag store → `cudaSleep(10000000000)`
  - Phase 2: Write loop + block flag store → sleep
  - Phase 3: Write loop + device flag store → sleep
  - Phase 4: Write loop + system flag store → sleep

```c
__device__ void gpu_buffer_writer_propagation_hierarchy_cpu(...)
```
- **Heterogeneous writer** (CPU + GPU readers)
- Waits for `CPU_NUM_THREADS + GPU_NUM_THREADS - 1` readers
- **TIMING:** Same structure as homogeneous writer

#### CPU Reader Functions (Templated)

**⏱️ TIMING INSERTION POINT:** Use `std::chrono::high_resolution_clock`

```c
template <typename B, typename R, typename W, typename F>
void cpu_buffer_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results, R *r_signal, W *w_signal, F *fallback_signal
)
```
- **TIMING:** Start when `w_signal` acquired, end after buffer read
- Acquire semantics on flag load

```c
template <typename B, typename R, typename W, typename F>
void cpu_buffer_reader_propagation_hierarchy_rlx(...)
```
- Relaxed flag load variant

```c
template <typename B, typename R, typename W, typename F>
void cpu_buffer_multi_reader_propagation_hierarchy_acq(...)
```
- Waits on all four scope flags
- **TIMING:** Track which flag succeeds first

```c
template <typename B, typename R, typename W, typename F>
void cpu_buffer_multi_reader_propagation_hierarchy_rlx(...)
```
- Relaxed multi-flag variant

#### CPU Writer Functions

**⏱️ TIMING INSERTION POINT:** Use `std::chrono::high_resolution_clock`

```c
void cpu_buffer_writer_propagation_hierarchy(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)
```
- **Homogeneous CPU writer**
- Waits for `CPU_NUM_THREADS - 1` readers
- Uses `sleep(5)` between phases (5 seconds)
- **TIMING:** Same structure as GPU writer (write + flag store for each scope)

```c
void cpu_buffer_writer_propagation_hierarchy_gpu(...)
```
- **Heterogeneous writer** (GPU + CPU readers)
- Waits for `CPU_NUM_THREADS + GPU_NUM_THREADS - 1` readers
- All flags use `memory_order_relaxed` (differs from homogeneous)
- **NAMING ISSUE:** Suffix suggests GPU involvement, but this is CPU-side writer for heterogeneous tests

#### Orchestration Functions

These dispatch to specific reader/writer variants based on thread/block ID.

```c
__global__ void gpu_buffer_reader_writer_propagation_hierarchy(
    bufferElement *buffer, bufferElement *w_buffer, bufferElement_na *results,
    flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal, WriterType *spawn_writer
)
```
- Block 0, Thread 0: Writer (if `spawn_writer != CE_NO_WRITER`)
- Other threads: Readers with scope/acquire pattern based on `threadIdx.x % 8`
  - 0-3: Relaxed readers (thread/block/device/system scopes)
  - 4-7: Acquire readers (or dummy readers if `NO_ACQ` defined)
- **TIMING:** Individual consumer functions handle timing

```c
void cpu_buffer_reader_writer_propagation_hierarchy(...)
```
- CPU thread dispatcher
- Core ID 0: Writer (if `spawn_writer != CE_NO_WRITER`)
- Other cores: Readers based on `core_id % 8`
- Uses `pthread_setaffinity_np` for core pinning (cores 32-63)

---

### Pattern 3: Propagation Hierarchy (Multi-Writer)

**Multi-producer mode (`-p` flag).** Four concurrent writers, each operating on a separate buffer with its own scope-level atomics. Tests whether multiple propagation paths interfere.

#### Buffer Organization

- `buffer_t` - Thread-scoped atomics
- `buffer_b` - Block-scoped atomics
- `buffer_d` - Device-scoped atomics
- `buffer_s` - System-scoped atomics
- `dummy_buffer` - Used by dummy readers as background load

#### GPU Multi-Writer Functions

**⏱️ TIMING INSERTION POINT:** Time write operation + flag store

```c
__device__ void gpu_buffer_multi_writer_thread_propagation_hierarchy(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
)
```
- Waits for `GPU_NUM_THREADS * GPU_NUM_BLOCKS - 4` readers
- Writes value 10, sleeps, writes value 1
- Thread scope writer
- **TIMING:** Write loop + `P_H_FLAG_STORE_ORDER` flag store

```c
__device__ void gpu_buffer_multi_writer_block_propagation_hierarchy(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
)
```
- Writes value 20, sleeps, writes value 2
- Block scope writer
- **TIMING:** Same as thread variant

```c
__device__ void gpu_buffer_multi_writer_device_propagation_hierarchy(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
)
```
- Writes value 30, sleeps, writes value 3
- Device scope writer

```c
__device__ void gpu_buffer_multi_writer_system_propagation_hierarchy(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
)
```
- Writes value 40, sleeps, writes value 4
- System scope writer

```c
__device__ void gpu_buffer_multi_writer_*_propagation_hierarchy_cpu(...)
```
- Heterogeneous variants (4 functions, one per scope)
- Wait for `CPU_NUM_THREADS + GPU_NUM_THREADS - 4` readers
- Otherwise identical to homogeneous variants

#### CPU Multi-Writer Functions

**⏱️ TIMING INSERTION POINT:** Use `std::chrono::high_resolution_clock`

```c
void cpu_buffer_multi_writer_thread_propagation_hierarchy(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
)
```
- Waits for `CPU_NUM_THREADS - 4` readers
- Writes value 10, sleeps 5s, writes value 1
- **TIMING:** Write loop + flag store for thread scope

Three additional variants for block/device/system scopes (values 20/2, 30/3, 40/4).

```c
void cpu_buffer_multi_writer_*_propagation_hierarchy_gpu(...)
```
- Heterogeneous variants (4 functions)
- Wait for `CPU_NUM_THREADS + GPU_NUM_THREADS - 4` readers
- **NAMING ISSUE:** Suffix `_gpu` suggests GPU code, but these are CPU functions for heterogeneous tests

#### Multi-Writer Orchestration

```c
__global__ void gpu_buffer_reader_multi_writer_propagation_hierarchy(
    bufferElement *dummy_buffer,
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement_na *results, flag_d *r_signal,
    flag_t *w_signal_t, flag_b *w_signal_b, flag_d *w_signal_d, flag_s *w_signal_s,
    flag_s *fb_signal, WriterType *spawn_writer
)
```
- Blocks 0-3, Thread 0: Four writers (one per scope)
- Other threads: Readers partitioned by `global_tid % 8`
  - Even 4 positions: Relaxed readers (one per scope level)
  - Odd 4 positions: Acquire readers (or dummy readers if `NO_ACQ`)
- **TIMING:** Individual writers/readers handle timing

```c
void cpu_buffer_reader_multi_writer_propagation_hierarchy(...)
```
- CPU dispatcher for multi-writer mode
- Cores 0, 8, 16, 24: Four writers (one per scope)
- Other cores: Readers based on `core_id % 8`

#### Dummy Workers (Background Load)

```c
__device__ void gpu_dummy_writer_worker_propagation(bufferElement *buffer, flag_d *r_signal)
```
- Background GPU write load (no synchronization with readers)
- **NO TIMING NEEDED** - Just background activity

```c
__device__ void gpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, flag_d *r_signal)
```
- Background GPU read load
- **NO TIMING NEEDED**

```c
template <typename R>
void cpu_dummy_reader_worker_propagation(bufferElement *buffer, bufferElement_na *results, R *r_signal)
```
- Background CPU read load
- **NO TIMING NEEDED**

---

## Utility Functions

### GPU Busy-Wait

```c
__device__ void cudaSleep(clock_t sleep_cycles)
```
- GPU-side delay using arithmetic busy-wait
- **Purpose:** Artificial delay between writer flag updates
- Uses `clock64()` for timing
- Performs arithmetic operations to prevent optimization
- **TIMING CONSIDERATION:** This introduces known delays; don't time this function itself

### Device Query

```c
int get_gpu_properties()
```
- Returns GPU clock rate (KHz)
- All other device properties are queried but output is commented out
- **Returns:** `prop.clockRate`
- Used for calculating time scales (though currently unused in main)

### Legacy Trigger

```c
__global__ void gpuTrigger(bufferElement *buffer, DATA_SIZE num, int chunkSize)
```
- **REDUNDANT** - Not used in current codebase

---

## Main Execution Flow

**File:** `cache_invalidation_testing.cu`

### Argument Parsing

```c
int main(int argc, char *argv[])
```

Parses command-line flags using `getopt`:
- `-m <allocator>` → `allocator_t` (AllocatorType)
- `-r <gpu|cpu>` → `reader_t` (ReaderWriterType)
- `-w <gpu|cpu>` → `writer_t` (ReaderWriterType)
- `-p` → `multi_producer` (bool)

### Memory Allocation

Based on `allocator_t`, allocates:

**Single-Writer Mode:**
- `buffer` - Main data buffer (`bufferElement * BUFFER_SIZE`)

**Multi-Writer Mode (`-p`):**
- `buffer_g_t/b/d/s` - Four scope-specific buffers
- `r_signal` - Reader readiness counter
- `w_signal_t/b/d/s` - Four writer signals (one per scope)
- `w_signal_fb` - Fallback timeout signal
- `dummy_buffer` - Background load buffer
- `result_g` - GPU result array (`GPU_NUM_BLOCKS * GPU_NUM_THREADS`)
- `result_c` - CPU result array (`CPU_NUM_THREADS`)

**Key Logic:** CUDA-only allocators (`cuda_malloc`) incompatible with CPU consumers.

### Execution Paths

#### Single-Writer Mode (Normal)

```c
if (reader == CE_GPU && writer == CE_GPU) {
    // GPU-only: Launch kernel with CE_WRITER
    gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(...);
    cudaDeviceSynchronize();
} else if (reader == CE_CPU && writer == CE_CPU) {
    // CPU-only: Launch CPU_NUM_THREADS threads with CE_WRITER
    // Core affinity: cores 32-63
} else {
    // Heterogeneous: Launch both GPU kernel and CPU threads
    // Writer on appropriate device (CE_HET_WRITER or CE_NO_WRITER)
}
```

#### Multi-Writer Mode (`-p` flag)

Similar structure but uses:
- `gpu_buffer_reader_multi_writer_propagation_hierarchy` (GPU)
- `cpu_buffer_reader_multi_writer_propagation_hierarchy` (CPU)
- Four separate buffers with different atomic scopes

### Result Collection

Results copied from device/thread-local storage to host and printed:
- GPU results: 2D array indexed by `[block][thread]`
- CPU results: 1D array indexed by core ID
- Values > 2,000,000,000 printed as `--` (indicates timeout/error)
- Result encoding reveals which scope flag succeeded first

### Memory Cleanup

Appropriate deallocation based on allocator type:
- System malloc → `free()`
- CUDA malloc/managed → `cudaFree()`
- CUDA host → `cudaFreeHost()`
- NUMA → `numa_free()`

---

## Build System

**File:** `Makefile`

### Targets

- `all` - Builds all flag/scope/size combinations
- `flag-rel` - Release-ordered flag variants only
- `flag-rlx` - Relaxed-ordered flag variants only
- `no-acq-flag-rel` - No-acquire + release flags
- `no-acq-flag-rlx` - No-acquire + relaxed flags
- `clean` - Remove all `.out` and `.ptx` files

### Compiler Configuration

```makefile
NVCC = nvcc
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
LIBS = -lnuma -lm
```

- `-g` - Debug symbols
- `-Xcompiler -O0` - Disable host compiler optimizations (preserves timing accuracy)
- `-Xcicc -O2` - CUDA intermediate compiler optimizations
- `-arch=sm_87` - Target architecture (adjust for your GPU: sm_80 for A100, sm_89 for RTX 4090, etc.)
- `-lnuma` - NUMA library for node-aware allocation
- `-lm` - Math library

### Build Matrix

Makefile generates targets for combinations of:
- **Scopes:** `CUDA_THREAD_SCOPE_THREAD` (default in current config)
- **Data sizes:** `DATA_SIZE_32` (default in current config)
- **Buffer modes:** `BUFFER_SAME` (currently used)
- **Flag orders:** `rel` (release) and `rlx` (relaxed)
- **Acquire modes:** Normal and `NO_ACQ`

**Output naming:** `cache_invalidation_testing_<flag>_<scope>_<size>_<buffer>.out`

**Example:**
- `cache_invalidation_testing_rel_CUDA_THREAD_SCOPE_THREAD_DATA_SIZE_32_BUFFER_SAME.out`
- `cache_invalidation_testing_no_acq_rlx_CUDA_THREAD_SCOPE_THREAD_DATA_SIZE_32_BUFFER_SAME.out`

To build for multiple scopes/sizes, modify `SCOPES` and `SIZES` variables in Makefile.

---

## Timing Implementation Guide

### For cudaEvent Timing (GPU Functions)

Create events before kernel launch and record around consumer execution:

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// Consumer kernel call
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

**Challenges:**
- Events measure kernel-level timing, not per-thread timing
- For per-thread timing, use `clock64()` inside device functions
- Store timing results in device memory arrays

**Target Functions:**
- All `gpu_buffer_reader_propagation_hierarchy_*` variants
- All `gpu_buffer_multi_reader_propagation_hierarchy_*` variants
- All `gpu_buffer_writer_propagation_hierarchy*` variants
- All `gpu_buffer_multi_writer_*_propagation_hierarchy*` variants

### For std::chrono Timing (CPU Functions)

Wrap consumer calls:

```cpp
auto start = std::chrono::high_resolution_clock::now();
// Consumer function call
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
```

**Target Functions:**
- All `cpu_buffer_reader_propagation_hierarchy_*` variants
- All `cpu_buffer_multi_reader_propagation_hierarchy_*` variants
- All `cpu_buffer_writer_propagation_hierarchy*` variants
- All `cpu_buffer_multi_writer_*_propagation_hierarchy*` variants

### What NOT to Time

- Dummy workers (`*_dummy_*`) - These are background load only
- Deprecated simple readers/writers - Not used in current tests
- `cudaSleep` / `sleep()` calls - These are intentional delays
- Memory allocation/deallocation - Not part of core measurement
- Result printing and collection - Post-test activities
- Orchestration functions - These just dispatch to actual consumers

### Timing Output Recommendations

Store timing data in per-thread/per-core arrays:
- GPU: `clock_t timing_data[GPU_NUM_BLOCKS * GPU_NUM_THREADS]`
- CPU: `std::chrono::nanoseconds timing_data[CPU_NUM_THREADS]`

Include in output:
- Thread/core ID
- Which scope flag was being waited on
- Acquire vs. relaxed semantics
- Timing value
- Result value (for verification)

---

## Function Status Quick Reference

| Function Pattern | Status | Timing Required | Notes |
|-----------------|--------|-----------------|-------|
| `gpu_buffer_reader_propagation_hierarchy_*` | ✓ Active | Yes | Core GPU readers |
| `cpu_buffer_reader_propagation_hierarchy_*` | ✓ Active | Yes | Core CPU readers |
| `gpu_buffer_multi_reader_propagation_hierarchy_*` | ✓ Active | Yes | Multi-flag readers |
| `cpu_buffer_multi_reader_propagation_hierarchy_*` | ✓ Active | Yes | Multi-flag CPU readers |
| `gpu_buffer_writer_propagation_hierarchy*` | ✓ Active | Yes | Core GPU writers |
| `cpu_buffer_writer_propagation_hierarchy*` | ✓ Active | Yes | Core CPU writers |
| `gpu_buffer_multi_writer_*_propagation_hierarchy*` | ✓ Active | Yes | Multi-writer GPU (4 variants) |
| `cpu_buffer_multi_writer_*_propagation_hierarchy*` | ✓ Active | Yes | Multi-writer CPU (4 variants) |
| `*_dummy_*_worker*` | ✓ Active | No | Background load only |
| `gpu_buffer_reader_writer_propagation_hierarchy` | ✓ Active | No | Orchestrator only |
| `cpu_buffer_reader_writer_propagation_hierarchy` | ✓ Active | No | Orchestrator only |
| `gpu_buffer_reader_multi_writer_propagation_hierarchy` | ✓ Active | No | Multi-writer orchestrator |
| `cpu_buffer_reader_multi_writer_propagation_hierarchy` | ✓ Active | No | Multi-writer orchestrator |
| `gpu_buffer_reader*` (simple) | ✗ Redundant | No | Legacy testing code |
| `gpu_buffer_writer*` (simple) | ✗ Redundant | No | Legacy testing code |
| `cpu_buffer_reader*` (simple) | ✗ Redundant | No | Legacy testing code |
| `cpu_buffer_writer*` (simple) | ✗ Redundant | No | Legacy testing code |
| `cudaSleep` | ✓ Utility | No | Intentional delay |
| `get_gpu_properties` | ✓ Utility | No | Device query |
| `gpuTrigger` | ✗ Redundant | No | Unused |

---

## Quick Function Lookup

### By Execution Pattern

**Single Writer, Homogeneous:**
- GPU: `gpu_buffer_writer_propagation_hierarchy`
- CPU: `cpu_buffer_writer_propagation_hierarchy`

**Single Writer, Heterogeneous:**
- GPU: `gpu_buffer_writer_propagation_hierarchy_cpu`
- CPU: `cpu_buffer_writer_propagation_hierarchy_gpu`

**Multi-Writer (4 concurrent):**
- GPU Thread: `gpu_buffer_multi_writer_thread_propagation_hierarchy[_cpu]`
- GPU Block: `gpu_buffer_multi_writer_block_propagation_hierarchy[_cpu]`
- GPU Device: `gpu_buffer_multi_writer_device_propagation_hierarchy[_cpu]`
- GPU System: `gpu_buffer_multi_writer_system_propagation_hierarchy[_cpu]`
- CPU variants: Same pattern with `cpu_` prefix

**Readers (All patterns):**
- Single flag wait: `*_reader_propagation_hierarchy_{acq|rlx}`
- Multi-flag wait: `*_multi_reader_propagation_hierarchy_{acq|rlx}`

### By Memory Ordering

**Acquire Readers:**
- `*_reader_propagation_hierarchy_acq`
- `*_multi_reader_propagation_hierarchy_acq`

**Relaxed Readers:**
- `*_reader_propagation_hierarchy_rlx`
- `*_multi_reader_propagation_hierarchy_rlx`

**Writers:**
- Flag store order controlled by compile-time `P_H_FLAG_STORE_ORDER_*` macros
- Release: `-DP_H_FLAG_STORE_ORDER_REL`
- Relaxed: `-DP_H_FLAG_STORE_ORDER_RLX`
