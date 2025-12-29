# Technical Reference

Complete technical reference for the CUDA Cache Invalidation Penalty Testing framework. This document details the current pattern-based dispatch system, data structures, and configuration format.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration System](#configuration-system)
3. [Data Structures](#data-structures)
4. [Pattern Dispatch](#pattern-dispatch)
5. [Consumer Functions](#consumer-functions)
6. [Build System](#build-system)

---

## System Overview

### Architecture

The framework uses a **pattern-based dispatch system** where test configurations are defined in YAML files. Each pattern specifies:
- Thread roles (writer, reader, dummy_reader, dummy_writer, inactive)
- Memory ordering semantics (acquire, release, relaxed, acq_rel)
- Thread scopes (thread, block, device, system)
- Which flag scope each reader watches

### Key Concepts

**Pattern-Based Testing:** All test configurations are defined in YAML files under `configs/`. The system loads patterns at runtime and dispatches threads to the appropriate consumer functions based on their role.

**Per-Thread Configuration:** Every thread (GPU and CPU) has an individual configuration specifying its exact behavior, enabling fine-grained control over cache coherence experiments.

**Multi-Writer Mode:** Patterns can enable multi-writer mode (`multi_writer: true`) which allocates 4 scope-specific buffers for concurrent writers.

---

## Configuration System

### YAML Pattern Format

```yaml
patterns:
  - name: "pattern_name"
    description: "Human-readable description"
    multi_writer: false  # Optional, default false
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        # Individual block configuration
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: reader, ordering: acquire, watch_flag: device}
        
        # Range configuration
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      threads:
        all_threads: {role: inactive}
```

### Thread Configuration Fields

#### role (Required)
- `writer` - Writes to buffer and sets flag
- `reader` - Waits on flag, reads buffer
- `dummy_writer` - Background traffic (writes to separate buffer)
- `dummy_reader` - Background traffic (reads from separate buffer)
- `inactive` - Thread does nothing

#### scope (Required for writer, optional for reader)
- `thread` - Thread-scope atomics
- `block` - Block-scope atomics
- `device` - Device-scope atomics
- `system` - System-scope atomics

#### ordering (Required for writer/reader)
- `relaxed` - memory_order_relaxed
- `acquire` - memory_order_acquire (readers only)
- `release` - memory_order_release (writers only)
- `acq_rel` - memory_order_acq_rel (not commonly used)

#### watch_flag (Required for readers)
Specifies which flag scope the reader waits on:
- `thread` - Wait on thread-scope flag
- `block` - Wait on block-scope flag
- `device` - Wait on device-scope flag
- `system` - Wait on system-scope flag

### Multi-Writer Mode

When `multi_writer: true`:
- Exactly 4 writers required (one per scope: thread, block, device, system)
- System allocates 4 scope-specific buffers (`bufferElement_t/b/d/s`)
- Writers automatically routed to their scope-specific buffer
- Readers routed to buffer matching their `watch_flag` scope

**Validation Rules:**
1. Exactly 4 writers total
2. Exactly 1 writer per scope
3. All readers must watch a scope that has a corresponding writer

### Command-Line Interface

```bash
# List available patterns
./output/cache_invalidation_testing_DATA_SIZE_32.out -F configs/pattern_file.yaml

# Run specific pattern
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -P pattern_name \
    -F configs/pattern_file.yaml \
    -m <allocator_type>
```

**Required Arguments:**
- `-P <pattern_name>` - Name of pattern to execute
- `-F <pattern_file>` - Path to YAML configuration file

**Optional Arguments:**
- `-m <allocator>` - Memory allocator (default: malloc)
  - `malloc` - System malloc
  - `numa_host` - NUMA allocation on CPU node
  - `numa_device` - NUMA allocation on GPU node
  - `dram` - CUDA pinned host memory (cudaMallocHost)
  - `um` - CUDA unified memory (cudaMallocManaged)
  - `cuda_malloc` - GPU device memory (GPU-only)

---

## Data Structures

### Buffer Elements

Each buffer element contains an atomic variable padded to PAGE_SIZE (4KB) for cache line isolation.

```c
typedef struct bufferElement {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_system> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement;
```

**Scope-Specific Variants (for multi-writer mode):**
```c
typedef struct bufferElement_t {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_t;

typedef struct bufferElement_b {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_block> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_b;

typedef struct bufferElement_d {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_device> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_d;

typedef struct bufferElement_s {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_system> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_s;

typedef struct bufferElement_na {
    uint32_t data;  // Non-atomic, for results
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} bufferElement_na;
```

### Synchronization Flags

```c
typedef struct flag_t {
    cuda::atomic<uint32_t, cuda::thread_scope_thread> flag;
    char padding[PAGE_SIZE - sizeof(uint32_t)];
} flag_t;

// flag_b, flag_d, flag_s follow same pattern
```

**Usage:**
- `r_signal` (flag_d) - Reader readiness counter
- `w_t_signal`, `w_b_signal`, `w_d_signal`, `w_s_signal` - Writer completion flags
- `fallback_signal` (flag_s) - Timeout mechanism

### Pattern Configuration Structures

```cpp
// Per-thread configuration (4 bytes)
struct ThreadConfig {
    ThreadRole role;           // 1 byte
    ThreadScope scope;         // 1 byte
    MemoryOrdering ordering;   // 1 byte
    ThreadScope watch_flag;    // 1 byte
};

// Pattern storage
struct PatternConfig {
    std::string name;
    std::string description;
    bool multi_writer;
    
    ThreadConfig gpu_threads[8][64];  // [block][thread]
    ThreadConfig cpu_threads[32];     // [core]
    
    int gpu_num_blocks;           // = 8
    int gpu_threads_per_block;    // = 64
    int cpu_num_threads;          // = 32
};
```

### Constants

```c
#define BUFFER_SIZE 512          // Elements per buffer
#define NUM_ITERATIONS 10000     // Iterations per consumer
#define GPU_NUM_BLOCKS 8         // GPU blocks
#define GPU_NUM_THREADS 64       // Threads per block
#define CPU_NUM_THREADS 32       // CPU threads
#define PAGE_SIZE 4096           // Padding size
```

---

## Pattern Dispatch

### GPU Dispatch Flow

```
pattern_orchestrator kernel
  └─> dispatch_gpu_thread(bid, tid, ...)
        └─> Read d_pattern_gpu[bid][tid]
        └─> Switch on role:
              ├─> WRITER: dispatch_writer(scope, ordering)
              ├─> READER: dispatch_reader(ordering, watch_flag)
              ├─> DUMMY_WRITER: gpu_dummy_writer(...)
              ├─> DUMMY_READER: gpu_dummy_reader(...)
              └─> INACTIVE: return
```

### Multi-Writer GPU Dispatch

```
pattern_orchestrator_multi kernel
  └─> dispatch_gpu_thread_multi(bid, tid, ...)
        └─> Read d_pattern_gpu[bid][tid]
        └─> Switch on role:
              ├─> WRITER: dispatch_multi_writer(scope, ordering)
              │             └─> Route to scope-specific buffer
              ├─> READER: dispatch_multi_reader(ordering, watch_flag)
              │             └─> Route to buffer matching watch_flag
              ├─> DUMMY_WRITER: gpu_dummy_writer(...)
              └─> DUMMY_READER: gpu_dummy_reader(...)
```

### CPU Dispatch Flow

```
CPU thread function
  └─> Read g_active_pattern->cpu_threads[core_id]
  └─> Switch on role:
        ├─> WRITER: dispatch_cpu_writer(scope, ordering)
        ├─> READER: dispatch_cpu_reader(ordering, watch_flag)
        ├─> DUMMY_READER: cpu_dummy_reader(...)
        └─> INACTIVE: return
```

### Device Constant Memory

Pattern configuration is copied to device constant memory for efficient access:

```cuda
__constant__ ThreadConfig d_pattern_gpu[8][64];

// In main():
cudaMemcpyToSymbol(d_pattern_gpu, pattern->gpu_threads,
                   sizeof(ThreadConfig) * 8 * 64);
```

---

## Consumer Functions

### GPU Writer Functions

#### Single-Writer with Release Ordering
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_writer_release(
    B *buffer,              // Buffer to write
    R *r_signal,            // Reader readiness counter
    W *w_signal,            // Writer completion flag
    flag_s *fallback_signal // Timeout flag
)
```

**Behavior:**
1. Wait for all readers: `while (r_signal < expected_readers)`
2. Write to buffer: `buffer[i].data.store(value, relaxed)`
3. Set flag: `w_signal->flag.store(1, memory_order_release)`
4. Sleep: `cudaSleep(10B cycles)`

#### Single-Writer with Relaxed Ordering
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_writer_relaxed(...)
```

Same behavior but uses `memory_order_relaxed` for flag store.

#### Multi-Writer Variants (4 scope levels × 2 orderings = 8 functions)
```cuda
__device__ void gpu_buffer_multi_writer_thread_release(...)
__device__ void gpu_buffer_multi_writer_thread_relaxed(...)
__device__ void gpu_buffer_multi_writer_block_release(...)
__device__ void gpu_buffer_multi_writer_block_relaxed(...)
__device__ void gpu_buffer_multi_writer_device_release(...)
__device__ void gpu_buffer_multi_writer_device_relaxed(...)
__device__ void gpu_buffer_multi_writer_system_release(...)
__device__ void gpu_buffer_multi_writer_system_relaxed(...)
```

**Multi-Writer Behavior:**
- Wait for `GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4` readers (4 concurrent writers)
- Write scope-specific value (thread=1, block=2, device=3, system=4)
- Set scope-specific flag

### GPU Reader Functions

#### Reader with Acquire Ordering
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_acquire(
    B *buffer,
    bufferElement_na *results,
    R *r_signal,
    W *w_signal,
    flag_s *fallback_signal
)
```

**Behavior:**
1. Optional pre-cache: Read buffer (if CONSUMERS_CACHE defined)
2. Signal ready: `r_signal->flag.fetch_add(1, relaxed)`
3. Wait on flag: `while (w_signal->flag.load(memory_order_acquire) == 0)`
4. Read buffer: Sum all elements
5. Store result: `results[tid].data = sum`

#### Reader with Relaxed Ordering
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_relaxed(...)
```

Same behavior but uses `memory_order_relaxed` for flag load.

### CPU Writer Functions

```cpp
template <typename B, typename W, typename R>
void cpu_buffer_writer_release(
    B *buffer, R *r_signal, W *w_signal, flag_s *fallback_signal
)
```

CPU writers follow same pattern as GPU writers but use:
- `std::this_thread::sleep_for()` instead of `cudaSleep()`
- C++ standard library atomics

### CPU Reader Functions

```cpp
template <typename B, typename R, typename W, typename F>
void cpu_buffer_reader_acquire(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, F *fallback_signal
)
```

CPU readers follow same pattern as GPU readers.

### Dummy Functions

**Purpose:** Generate background memory traffic without participating in synchronization.

```cuda
__device__ void gpu_dummy_writer(bufferElement *dummy_buffer)
__device__ void gpu_dummy_reader(bufferElement *dummy_buffer, bufferElement_na *results)
```

```cpp
void cpu_dummy_reader(bufferElement *dummy_buffer, bufferElement_na *results)
```

**Behavior:**
- Loop for NUM_ITERATIONS (10,000)
- Read/write dummy_buffer with relaxed ordering
- No flag synchronization

---

## Build System

### Makefile Targets

```bash
make all    # Build all data size variants
make clean  # Remove all build artifacts
```

### Build Variants

The system builds 4 executables, one per data size:

```
output/cache_invalidation_testing_DATA_SIZE_8.out   # uint8_t
output/cache_invalidation_testing_DATA_SIZE_16.out  # uint16_t
output/cache_invalidation_testing_DATA_SIZE_32.out  # uint32_t
output/cache_invalidation_testing_DATA_SIZE_64.out  # uint64_t
```

### Compiler Flags

```makefile
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87 -Iinclude
LIBS = -lnuma -lm
```

**Key Defines:**
- `PATTERN_DISPATCH` - Enables pattern-based dispatch system
- `DATA_SIZE_*` - Selects data type (8/16/32/64-bit)
- `CONSUMERS_CACHE` - Enables pre-caching in readers (currently always on)

### Architecture Target

Currently targets **NVIDIA Hopper (sm_87)**. Modify `-arch=sm_87` for other architectures.

---

## File Structure

```
include/
  ├── types.hpp               # Data structures, enums, constants
  ├── pattern_config.hpp      # Pattern configuration structures
  ├── pattern_dispatch.cuh    # GPU consumer functions and dispatch
  └── pattern_dispatch_cpu.hpp # CPU consumer functions and dispatch

src/
  ├── cache_invalidation_testing.cu  # Main entry point
  └── pattern_config.cpp             # YAML parsing and validation

configs/
  ├── isolated_acquire.yaml          # Isolated acquire tests
  ├── test_multi_writer.yaml         # Multi-writer tests
  └── ...                            # Additional patterns

docs/
  ├── REFERENCE.md              # This file
  ├── MULTI_WRITER.md          # Multi-writer system documentation
  ├── REDUNDANT.md             # Deprecated features
  └── TODO.md                  # Cleanup and enhancement tasks

stale_code.cuh                 # Legacy functions (not used)
```

---

## Example Pattern

```yaml
patterns:
  - name: "basic_release_acquire"
    description: "Single writer with release, readers with acquire"
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_31: {role: reader, ordering: acquire, watch_flag: device}
          threads_32_63: {role: dummy_reader}
        
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      threads:
        all_threads: {role: inactive}
```

**Execution:**
1. Thread 0 in Block 0 writes to buffer, sets device-scope flag with release
2. Threads 1-31 in Block 0 wait on device flag with acquire, read buffer
3. Threads 32-63 in Block 0 + all threads in Blocks 1-7 generate dummy traffic
4. All CPU threads are inactive

---

## Validation

The pattern registry validates patterns on load:

1. **Role Validation:** All required fields present
2. **Writer Count:** At least one writer exists
3. **Multi-Writer Validation:** If `multi_writer: true`:
   - Exactly 4 writers
   - Exactly 1 writer per scope
   - All readers watch valid scopes
4. **Flag Coverage:** All readers watch flags that will be set

Invalid patterns are rejected with detailed error messages.

---

## Legacy Information

For information about deprecated features (old CLI flags, propagation hierarchy functions, etc.), see [REDUNDANT.md](REDUNDANT.md).
