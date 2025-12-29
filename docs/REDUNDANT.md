# Documentation Update Summary

**Date:** December 29, 2025  
**Update Type:** Major Documentation Overhaul

## Overview

The codebase has undergone a major refactoring from a hard-coded thread assignment system to a flexible **pattern-based dispatch system** using YAML configuration files. This update brings the documentation in line with the current implementation.

---

## What Changed in the Codebase

### Major Architectural Changes

1. **Pattern-Based Configuration System**
   - YAML files define test patterns (`configs/*.yaml`)
   - Per-thread role, scope, and ordering specification
   - Runtime pattern loading and validation
   - Multi-writer mode integrated into pattern system

2. **Removed Command-Line Flags**
   - ❌ `-r <gpu|cpu>` (reader type)
   - ❌ `-w <gpu|cpu>` (writer type)
   - ❌ `-p` (multi-producer toggle)
   - ✅ `-P <pattern>` (pattern name)
   - ✅ `-F <file>` (YAML config file)

3. **Simplified Build System**
   - Only data size variants (8/16/32/64-bit)
   - No more ordering combination variants
   - Single build target: `make all`

4. **Legacy Code Movement**
   - Old propagation hierarchy functions → `stale_code.cuh`
   - Old orchestrators removed from main codebase
   - Hard-coded thread assignments eliminated

---

## Documentation Files Status

### ✅ New/Updated Files (CURRENT)

| File | Status | Description |
|------|--------|-------------|
| [REFERENCE_NEW.md](REFERENCE_NEW.md) | ✅ NEW | Complete reference for current pattern dispatch system |
| [REDUNDANT.md](REDUNDANT.md) | ✅ NEW | Historical documentation of deprecated features |
| [TODO_NEW.md](TODO_NEW.md) | ✅ NEW | Updated task list for current system |
| [MULTI_WRITER.md](MULTI_WRITER.md) | ✅ UPDATED | Updated usage examples for pattern-based multi-writer |
| [README.md](../README.md) | ✅ UPDATED | Complete rewrite for pattern dispatch system |

### ⚠️ Outdated Files (NEEDS REVIEW)

| File | Status | Issue | Recommendation |
|------|--------|-------|----------------|
| [REFERENCE.md](REFERENCE.md) | ⚠️ OUTDATED | Describes old propagation hierarchy | Replace with REFERENCE_NEW.md |
| [THREAD_ORGANIZATION.md](THREAD_ORGANIZATION.md) | ⚠️ OUTDATED | Documents hard-coded thread assignments | Archive or add "OUTDATED" notice |
| [TODO.md](TODO.md) | ⚠️ OUTDATED | Tasks for deprecated code | Replace with TODO_NEW.md |

---

## Key Documentation Updates

### README.md

**Before:**
```bash
# Old usage with deprecated flags
./cache_invalidation_testing -m malloc -r gpu -w cpu -p
```

**After:**
```bash
# New pattern-based usage
./cache_invalidation_testing_DATA_SIZE_32.out \
    -P pattern_name \
    -F configs/pattern_file.yaml \
    -m um
```

**New Sections Added:**
- Pattern-Based Configuration
- YAML Pattern Format
- Thread Roles and Configuration
- Multi-Writer Mode
- Example Patterns
- Documentation Status Notice

### REFERENCE_NEW.md

Complete rewrite covering:
- System architecture and concepts
- YAML configuration format
- Pattern validation rules
- Data structures
- Pattern dispatch flow
- Consumer function reference
- Build system
- File structure

### REDUNDANT.md

Documents deprecated features:
- Old command-line interface (-r, -w, -p)
- Propagation hierarchy functions
- Multi-writer orchestrator functions
- Old thread organization patterns
- Compile-time ordering flags
- Historical build targets
- Evolution of the codebase

### MULTI_WRITER.md

**Changes:**
- Updated overview to clarify current status
- Updated usage examples with new command syntax
- No changes needed to implementation details (still accurate)

---

## User Migration Guide

### For Users of Old System

If you have scripts using the old command-line interface:

**Old:**
```bash
./cache_invalidation_testing_acq_rel_CUDA_THREAD_SCOPE_DEVICE_DATA_SIZE_32.out \
    -m malloc -r gpu -w cpu
```

**New:**
```bash
# 1. Create a YAML pattern file (e.g., configs/my_pattern.yaml)
# 2. Run with new syntax
./cache_invalidation_testing_DATA_SIZE_32.out \
    -P my_pattern_name \
    -F configs/my_pattern.yaml \
    -m malloc
```

### Creating Equivalent Patterns

**Old Configuration:**
- `-r gpu -w gpu` (single GPU writer, GPU readers)
- Compile-time: `C_H_FLAG_LOAD_ORDER_ACQ`, `P_H_FLAG_STORE_ORDER_REL`

**Equivalent YAML Pattern:**
```yaml
patterns:
  - name: "gpu_writer_gpu_readers"
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: reader, ordering: acquire, watch_flag: device}
        blocks_1_7:
          all_threads: {role: dummy_reader}
```

---

## For Developers

### Where to Find Information

| Topic | Current Reference | Old Reference (Deprecated) |
|-------|-------------------|---------------------------|
| System overview | REFERENCE_NEW.md | REFERENCE.md |
| Pattern configuration | REFERENCE_NEW.md | N/A (new feature) |
| Consumer functions | REFERENCE_NEW.md | REFERENCE.md |
| Multi-writer | MULTI_WRITER.md | REFERENCE.md |
| Thread organization | REFERENCE_NEW.md (Pattern Dispatch) | THREAD_ORGANIZATION.md |
| Build system | REFERENCE_NEW.md, README.md | README.md (partially outdated) |
| Deprecated features | REDUNDANT.md | N/A |
| Future work | TODO_NEW.md | TODO.md |

### Code Navigation

**Current Active Code:**
```
src/cache_invalidation_testing.cu     # Main, pattern loading
src/pattern_config.cpp                 # YAML parsing
include/pattern_dispatch.cuh           # GPU dispatch + consumers
include/pattern_dispatch_cpu.hpp       # CPU consumers
include/pattern_config.hpp             # Configuration structures
include/types.hpp                      # Data types
```

**Legacy Code (Not Used):**
```
stale_code.cuh                        # Old consumer functions
```

---

## Testing the Documentation

### Recommended Review Process

1. **Build and Run:**
   ```bash
   make all
   ./output/cache_invalidation_testing_DATA_SIZE_32.out \
       -F configs/isolated_acquire.yaml
   ```

2. **Test Pattern Loading:**
   ```bash
   # Should list all patterns
   ./output/cache_invalidation_testing_DATA_SIZE_32.out \
       -F configs/test_multi_writer.yaml
   ```

3. **Run Multi-Writer:**
   ```bash
   ./output/cache_invalidation_testing_DATA_SIZE_32.out \
       -P multi_writer_test \
       -F configs/test_multi_writer.yaml \
       -m um
   ```

4. **Verify Documentation:**
   - Check that all examples in README.md work
   - Verify YAML syntax examples are valid
   - Confirm file paths in docs match actual structure

---

## Next Steps

### Immediate (Critical)

1. **Replace Old Files:**
   ```bash
   mv docs/REFERENCE.md docs/REFERENCE_OLD.md
   mv docs/REFERENCE_NEW.md docs/REFERENCE.md
   mv docs/TODO.md docs/TODO_OLD.md
   mv docs/TODO_NEW.md docs/TODO.md
   ```

2. **Add Deprecation Notices:**
   Add header to THREAD_ORGANIZATION.md:
   ```markdown
   # ⚠️ OUTDATED DOCUMENTATION
   
   This document describes a previous version of the system with hard-coded
   thread assignments. The current system uses YAML-based pattern configuration.
   
   See REDUNDANT.md for historical context.
   See REFERENCE.md for current system documentation.
   ```

### Short Term (High Priority)

3. **Create Pattern Guide:**
   - Document common patterns
   - Provide templates
   - Add troubleshooting section

4. **Add Examples:**
   - Create EXAMPLES.md with annotated patterns
   - Show progression from simple to complex

5. **Improve Error Messages:**
   - Better YAML validation errors
   - Helpful suggestions when patterns fail

### Long Term (Enhancement)

6. **Interactive Tools:**
   - Web-based pattern builder
   - Pattern visualization
   - Validation as you type

7. **Video Tutorials:**
   - Introduction to pattern system
   - Creating your first pattern
   - Advanced multi-writer patterns

---

## Validation Checklist

Before finalizing documentation update:

- [ ] All code examples compile and run
- [ ] All file paths are correct
- [ ] YAML examples are syntactically valid
- [ ] All cross-references between docs work
- [ ] README.md accurately reflects current CLI
- [ ] Build instructions produce working executables
- [ ] Example patterns run without errors
- [ ] Deprecation notices are clear
- [ ] Migration guide is complete
- [ ] File structure diagram matches reality

---

## Questions and Feedback

If you find documentation errors or unclear sections:

1. Check REDUNDANT.md to see if feature was deprecated
2. Verify against actual code in `src/` and `include/`
3. Test with actual pattern YAML files in `configs/`
4. File an issue or update documentation directly

---

## Acknowledgments

This documentation update preserves the detailed technical work in the original REFERENCE.md and THREAD_ORGANIZATION.md while accurately reflecting the current pattern-based system. Historical information has been moved to REDUNDANT.md rather than deleted, maintaining the project's evolution history.


# Redundant/Legacy Documentation

This document contains documentation for features that have been removed or replaced in the current codebase. These are preserved for historical reference.

---

## Deprecated Command-Line Interface

**Status:** REMOVED in current version

### Old Runtime Flags

The following command-line flags were used in previous versions but are no longer supported:

- `-r <gpu|cpu>` - Reader type selection
- `-w <gpu|cpu>` - Writer type selection  
- `-p` - Multi-producer mode toggle

### Old Usage Pattern

```bash
# DEPRECATED - No longer works
./cache_invalidation_testing -m malloc -r gpu -w cpu -p
```

**Replacement:** Use pattern-based YAML configuration with `-P` and `-F` flags.

---

## Deprecated Propagation Hierarchy Functions

**Status:** Code moved to `stale_code.cuh`

The following functions were the original implementation before pattern dispatch was introduced. They implemented a phased writer that progressively set flags at different scope levels.

### Simple Reader/Writer Functions (GPU)

All located in `stale_code.cuh`:

```c
// Single-iteration readers
__global__ void gpu_buffer_reader_single_iter(bufferElement *buffer, uint32_t *results, clock_t *duration)
__device__ void gpu_buffer_reader_diverge(bufferElement *buffer, uint32_t *results, uint32_t *duration)
__device__ void gpu_buffer_reader_diverge_constant(bufferElement *buffer, uint32_t *result)

// Multi-iteration readers
__global__ void gpu_buffer_reader(bufferElement *buffer, uint32_t *results, uint32_t *duration)

// Writers
__global__ void gpu_buffer_writer_single_iter(bufferElement *buffer, int chunkSize)
__global__ void gpu_buffer_writer(bufferElement *buffer, int chunkSize, clock_t *sleep_duration)
__device__ void gpu_buffer_writer_diverge(bufferElement *buffer, clock_t *sleep_duration)

// Combined
__global__ void gpu_buffer_reader_writer(bufferElement *buffer, bufferElement *w_buffer, ...)
```

### Propagation Hierarchy Reader Functions (GPU)

**Original Pattern:** One designated writer updated data at specific intervals, progressively setting flags for each thread scope level.

```c
// GPU readers with acquire semantics
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal
)

// GPU readers with relaxed semantics
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_propagation_hierarchy_rlx(...)

// Multi-flag readers (wait on all 4 scope flags)
template <typename B, typename W, typename R>
__device__ void gpu_buffer_multi_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results, R *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)
```

### Propagation Hierarchy Writer Functions (GPU)

```c
// Homogeneous writer (GPU-only readers)
__device__ void gpu_buffer_writer_propagation_hierarchy(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)

// Heterogeneous writer (CPU + GPU readers)
__device__ void gpu_buffer_writer_propagation_hierarchy_cpu(...)
```

**Writer Phasing Pattern:**
```
Phase 1: buffer[i] = 10 → flag_thread = 1   → cudaSleep(10B cycles)
Phase 2: buffer[i] = 20 → flag_block = 1    → cudaSleep(10B cycles)
Phase 3: buffer[i] = 30 → flag_device = 1   → cudaSleep(10B cycles)
Phase 4: buffer[i] = 40 → flag_system = 1   → cudaSleep(10B cycles)
Phase 5: buffer[i] = 50 → fallback = 4      → cudaSleep(10B cycles)
```

### CPU Propagation Hierarchy Functions

```c
// CPU readers
template <typename B, typename R, typename W, typename F>
void cpu_buffer_reader_propagation_hierarchy_acq(...)
void cpu_buffer_reader_propagation_hierarchy_rlx(...)
void cpu_buffer_multi_reader_propagation_hierarchy_acq(...)
void cpu_buffer_multi_reader_propagation_hierarchy_rlx(...)

// CPU writers
void cpu_buffer_writer_propagation_hierarchy(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
)
void cpu_buffer_writer_propagation_hierarchy_gpu(...)
```

### Orchestrator Functions

These dispatched to specific reader/writer variants based on thread/block ID:

```c
// GPU orchestrator
__global__ void gpu_buffer_reader_writer_propagation_hierarchy(
    bufferElement *buffer, bufferElement *w_buffer,
    bufferElement_na *results, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal, WriterType *spawn_writer
)

// CPU orchestrator
void cpu_buffer_reader_writer_propagation_hierarchy(...)
```

**Thread Assignment Pattern (Old):**
- Block 0, Thread 0: Writer (if spawn_writer != CE_NO_WRITER)
- Other threads: Readers based on threadIdx.x % 8
  - 0-3: Relaxed readers (thread/block/device/system scopes)
  - 4-7: Acquire readers (or dummy if NO_ACQ defined)

---

## Deprecated Multi-Writer Functions

**Status:** Replaced by pattern-based multi-writer mode

### Old Multi-Writer GPU Functions

```c
__device__ void gpu_buffer_multi_writer_thread_propagation_hierarchy(
    bufferElement_t *buffer, flag_d *r_signal, flag_t *w_signal, flag_s *fb_signal
)

__device__ void gpu_buffer_multi_writer_block_propagation_hierarchy(
    bufferElement_b *buffer, flag_d *r_signal, flag_b *w_signal, flag_s *fb_signal
)

__device__ void gpu_buffer_multi_writer_device_propagation_hierarchy(
    bufferElement_d *buffer, flag_d *r_signal, flag_d *w_signal, flag_s *fb_signal
)

__device__ void gpu_buffer_multi_writer_system_propagation_hierarchy(
    bufferElement_s *buffer, flag_d *r_signal, flag_s *w_signal, flag_s *fb_signal
)
```

**Old Pattern:** 4 concurrent writers, each with its own buffer and scope level.

### Old Multi-Writer Orchestrator

```c
__global__ void gpu_buffer_reader_multi_writer_propagation_hierarchy(
    bufferElement_t *buffer_t, bufferElement_b *buffer_b,
    bufferElement_d *buffer_d, bufferElement_s *buffer_s,
    bufferElement *dummy_buffer, bufferElement_na *results,
    flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal
)
```

**Thread Assignment (Old):**
- Blocks 0-3: Writers (one scope per block, thread 0)
- All other threads: Readers based on global_tid % 8
- Hard-coded block filters (bid == 5, bid == 99) limited actual reader count

---

## Deprecated Thread Organization

### Old GPU Configuration Patterns

**Single-Writer Mode Issues:**
- Hard-coded `bid == 6` and `bid == 99` filters
- Only ~10-15 actual readers out of 512 threads (~2-3%)
- Block 0: 1 writer + ~7 readers + ~56 dummy
- Block 6: 1 dummy writer + ~3 readers + ~60 dummy
- Blocks 1-5, 7: All dummy threads

**Multi-Producer Mode Issues:**
- Hard-coded `bid == 5` and `bid == 99` filters
- Uneven distribution of actual vs. dummy threads
- Complex modulo patterns (%8) limited flexibility

### Old CPU Thread Assignment

**Single-Writer Mode:**
- Core 32 (core_id % 32 == 0): Writer
- Cores 33-63: Readers based on core_id % 8

**Multi-Producer Mode:**
- Writers on cores 32, 40, 48, 56 (core_id % 8 == 0)
- Readers on other cores based on modulo patterns

---

## Deprecated Compile-Time Memory Ordering Flags

**Status:** Still present but less relevant with pattern dispatch

### Old Ordering Flags

```c
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
```

**Replacement:** Memory ordering is now specified per-thread in YAML patterns via the `ordering` field.

### Old Build Targets

```makefile
# DEPRECATED Makefile targets
make acq-rel    # Acquire-release variants
make acq-rlx    # Acquire-relaxed variants
make rlx-rel    # Relaxed-release variants
make rlx-rlx    # Relaxed-relaxed variants
```

**Replacement:** The current Makefile builds pattern-dispatch-enabled executables with data size variants only.

---

## Deprecated Configuration Space

### Old Experimental Space

**Compile-Time:** 4 (ordering) × 4 (scopes) × 4 (data sizes) = 64 variants  
**Runtime:** 6 (allocators) × 2 (reader types) × 2 (writer types) × 2 (multi-producer) = 96 configs  
**Total:** 6,144 possible configurations

**Current:** Pattern-based system with unlimited YAML configurations, 4 data size variants.

---

## Historical Context

### Evolution of the Codebase

1. **Phase 1 (Early):** Simple reader/writer functions with hard-coded behavior
2. **Phase 2 (Propagation Hierarchy):** Phased writers with scope progression
3. **Phase 3 (Multi-Writer):** Concurrent writers at different scope levels
4. **Phase 4 (Current - Pattern Dispatch):** YAML-based flexible configuration system

### Reasons for Deprecation

1. **Limited Flexibility:** Hard-coded thread assignments couldn't express desired test patterns
2. **Complex Maintenance:** Multiple orchestrator variants with subtle differences
3. **Poor Scalability:** Adding new patterns required code changes
4. **Configuration Complexity:** Compile-time + runtime flags created confusing combinations
5. **Documentation Burden:** Hard to explain and maintain thread organization logic

### Lessons Learned

- Configuration-driven testing is more maintainable than hard-coded patterns
- Per-thread control is essential for fine-grained cache coherence experiments
- YAML provides readable, version-controllable test configurations
- Separation of pattern logic from execution infrastructure improves code quality


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



# TODO: Code Cleanup and Timing Implementation

This document tracks necessary refactoring and the implementation plan for comprehensive timing instrumentation.

## Table of Contents
1. [Redundant/Unused Functions](#redundantunused-functions-to-remove)
2. [Inconsistent Code](#inconsistent-code-requiring-cleanup)
3. [Timing Implementation Plan](#timing-implementation-plan)

---

## Redundant/Unused Functions to Remove

These functions are not called in the main execution path and should be removed to reduce code bloat and confusion.

### Simple Reader Functions (GPU) - REMOVE ALL

**Location:** Lines ~270-410

```c
// REDUNDANT - Remove these
static void buffer_reader_single_iter(bufferElement * buffer)
__global__ void gpu_buffer_reader_single_iter(bufferElement *buffer, uint32_t *results, clock_t *duration)
__global__ void gpu_buffer_reader(bufferElement *buffer, uint32_t *results, uint32_t *duration)
__device__ void gpu_buffer_reader_diverge(bufferElement *buffer, uint32_t *results, uint32_t *duration)
__device__ void gpu_buffer_reader_diverge_constant(bufferElement *buffer, uint32_t *result)
```

**Reason:** All superseded by propagation hierarchy readers. Have built-in timing that won't be used.

### Simple Writer Functions (GPU) - REMOVE ALL

**Location:** Lines ~360-410

```c
// REDUNDANT - Remove these
__global__ void gpu_buffer_writer_single_iter(bufferElement *buffer, int chunkSize)
__global__ void gpu_buffer_writer_single_iter_single_thread(bufferElement *buffer, int chunkSize)
__global__ void gpu_buffer_writer(bufferElement *buffer, int chunkSize, clock_t *sleep_duration)
__device__ void gpu_buffer_writer_diverge(bufferElement *buffer, clock_t *sleep_duration)
__device__ void gpu_buffer_writer_diverge_constant(bufferElement *buffer)
__global__ void gpu_buffer_writer_single_thread(bufferElement *buffer, int chunkSize, clock_t *sleep_duration)
```

**Reason:** All superseded by propagation hierarchy writers.

### Combined Reader-Writer Functions (GPU) - REMOVE ALL

**Location:** Lines ~440-515

```c
// REDUNDANT - Remove these
__global__ void gpu_buffer_reader_writer(bufferElement *buffer, bufferElement *w_buffer, clock_t *sleep_duration, uint32_t *results, uint32_t *duration)
__global__ void gpu_buffer_reader_writer_constant(bufferElement *buffer, bufferElement *w_buffer, uint32_t *result, clock_t *t_reader, clock_t *t_writer)
```

**Reason:** These were early prototypes. Replaced by orchestrator functions.

### Simple CPU Functions - REMOVE ALL

**Location:** Lines ~519-600

```c
// REDUNDANT - Remove these
void cpu_buffer_writer_single_iter(bufferElement *buffer)
void cpu_buffer_writer(bufferElement *buffer, struct timespec *sleep_duration)
void cpu_buffer_reader_single_iter(bufferElement *buffer)
void cpu_buffer_reader(bufferElement *buffer, uint32_t *result, std::chrono::duration<uint32_t, std::nano> *duration)
void buffer_reader(bufferElement *buffer)  // Has hardcoded sleep(250)
```

**Reason:** All superseded by propagation hierarchy CPU functions.

### Utility Functions - REMOVE

**Location:** Line ~640

```c
// REDUNDANT - Remove this
__global__ void gpuTrigger(bufferElement *buffer, DATA_SIZE num, int chunkSize)
```

**Reason:** Never called anywhere in codebase.

### Summary of Removal Impact

- **Before:** ~2454 lines
- **After removal:** ~2000 lines (estimated)
- **Functions to remove:** 18 functions
- **Lines to remove:** ~450 lines

---

## Inconsistent Code Requiring Cleanup

These sections have commented-out code, random edits, or don't follow established patterns.

### 1. Commented Iterator Variable in Loop

**Location:** Lines 327, 348 (gpu_buffer_reader_diverge, gpu_buffer_writer_diverge)

```c
// INCONSISTENT - Wrong variable in comment
for (int i = 0; i < NUM_ITERATIONS; i++) {
    clock_t begin = clock64();
    // for (int k = 0; k < NUM_ITERATIONS / 100; i++) {  // <-- WRONG: should be k++, not i++
        for (int j = 0; j < BUFFER_SIZE; j++) {
```

**Action:** Remove commented line entirely (function is being removed anyway).

### 2. Commented Old Function Signatures

**Location:** Lines 519, 526, 539, 549

```c
// INCONSISTENT - Old signatures left as comments
// static void __attribute__((optimize("O0"))) cpu_buffer_writer_single_iter(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer) {
static void __attribute__((optimize("O0"))) cpu_buffer_writer_single_iter(bufferElement *buffer) {
```

**Action:** Remove commented old signatures when removing functions.

### 3. Large Block of Commented Generic Reader

**Location:** Lines ~601-635

```c
// INCONSISTENT - 35 lines of commented code
// static void __attribute__((optimize("O0"))) buffer_reader(DATA_SIZE *buffer) {
//     for (DATA_SIZE i = 0; i < NUM_ITERATIONS; i++) {
//         ...
//     }
// }
```

**Action:** Remove entire commented block.

### 4. Random Incomplete Comment

**Location:** Line 637

```c
// INCONSISTENT - Incomplete thought
// for (int )
```

**Action:** Remove this line.

### 5. Random Incomplete Loop Comment

**Location:** Line 645

```c
// INCONSISTENT - Weird negative loop condition comment
// for (int j = 0; j > -1; j++) {
```

**Action:** Remove this line.

### 6. Commented Iteration Loops in Active Functions

**Location:** Lines 699, propagation writer functions

```c
// INCONSISTENT - Why is iteration commented out?
// for (int i = 0; i < NUM_ITERATIONS; i++) {
    for (int j = 0; j < BUFFER_SIZE; j++) {
        buffer[j].data.store(10, cuda::memory_order_relaxed);
    }
// }
```

**Action:** Either remove comments or clarify intent. If single iteration is intentional, add comment explaining why.

### 7. Massive Block of Commented Device Properties

**Location:** Lines 235-267 (get_gpu_properties)

```c
// INCONSISTENT - 30+ lines of commented std::cout statements
// std::cout << "Device name: " << prop.name << std::endl;
// std::cout << "Total Global Memory: " << prop.totalGlobalMem << std::endl;
// ... 30 more lines ...
```

**Action:** 
- Option A: Remove all commented lines, keep only the return statement
- Option B: Create a verbose flag to enable full output
- **Recommended:** Option A (remove), since info can be queried with nvidia-smi

### 8. Commented printf Statements

**Location:** Lines 1027, 1107, and throughout

```c
// INCONSISTENT - Debug printfs left commented
// printf("[GPU] Writer Done\n");
// printf("[GPU] Het-Writer Done\n");
```

**Action:** Remove commented debug statements or implement proper debug flag system.

### 9. Commented Alternative Function Calls

**Location:** Lines 1139, 1148, 1151, 1154, 1157, 1188, 1195, 1202

```c
// INCONSISTENT - Old function calls left commented in orchestrators
// gpu_buffer_writer_propagation_hierarchy(buffer, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
// gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, w_t_signal, fallback_signal);
```

**Action:** Remove these commented calls. The code has evolved to use multi-reader versions.

### 10. Type Mismatch in bufferElement_d

**Location:** Line 147

```c
// BUG - Should be cuda::thread_scope_device, not thread_scope_thread
typedef struct bufferElement_d {
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;  // <-- WRONG SCOPE
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_d;
```

**Action:** Fix to `cuda::thread_scope_device`

---

## Timing Implementation Plan

Comprehensive plan to add timing instrumentation to all active consumer functions.

### Goals

1. Measure per-consumer timing for readers and writers
2. Separate timing for each phase of multi-phase writers
3. Identify which scope flag triggered first in multi-reader tests
4. Minimal performance impact from timing infrastructure
5. Clear, parseable output format for analysis

### Data Structures

#### 1. GPU Timing Structures

**Location:** Add near other structs (after flag definitions, ~line 190)

```c
/**
 * @brief Per-thread GPU timing data
 * 
 * Stored in device memory, copied to host after kernel completion.
 * Uses clock_t (clock64() return type) for GPU timing.
 */
typedef struct gpu_timing_data {
    clock_t start_time;           // When consumer started
    clock_t end_time;             // When consumer completed
    clock_t duration;             // end_time - start_time (computed on device)
    uint32_t flag_type;           // Which flag triggered (0=thread, 1=block, 2=device, 3=system)
    uint32_t iteration;           // For multi-iteration consumers
    uint32_t consumer_type;       // 0=reader_rlx, 1=reader_acq, 2=writer, 3=dummy
    uint32_t thread_id;           // Global thread ID for identification
    char padding[PAGE_SIZE - sizeof(clock_t)*3 - sizeof(uint32_t)*4];
} gpu_timing_data;

/**
 * @brief Multi-phase writer timing (4 phases)
 * 
 * For writers that set flags at 4 different scope levels.
 */
typedef struct gpu_writer_phase_timing {
    clock_t phase_start[4];       // Start time for each phase
    clock_t phase_end[4];         // End time for each phase
    clock_t phase_duration[4];    // Duration of each phase
    uint32_t thread_id;
    char padding[PAGE_SIZE - sizeof(clock_t)*12 - sizeof(uint32_t)];
} gpu_writer_phase_timing;
```

**Memory Allocation:**
- Allocate in **device memory** (cudaMalloc)
- Size: `GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data)`
- For multi-phase writers: `4 * sizeof(gpu_writer_phase_timing)` (one per scope level)

#### 2. CPU Timing Structures

**Location:** Add near GPU timing structs

```c
/**
 * @brief Per-thread CPU timing data
 * 
 * Stored in host memory, filled by each CPU thread.
 * Uses std::chrono::nanoseconds for CPU timing.
 */
typedef struct cpu_timing_data {
    uint64_t start_ns;            // Start time in nanoseconds since epoch
    uint64_t end_ns;              // End time in nanoseconds since epoch
    uint64_t duration_ns;         // Duration in nanoseconds
    uint32_t flag_type;           // Which flag triggered
    uint32_t iteration;           // For multi-iteration consumers
    uint32_t consumer_type;       // 0=reader_rlx, 1=reader_acq, 2=writer, 3=dummy
    uint32_t core_id;             // CPU core ID
    char padding[PAGE_SIZE - sizeof(uint64_t)*3 - sizeof(uint32_t)*4];
} cpu_timing_data;

/**
 * @brief Multi-phase CPU writer timing
 */
typedef struct cpu_writer_phase_timing {
    uint64_t phase_start_ns[4];
    uint64_t phase_end_ns[4];
    uint64_t phase_duration_ns[4];
    uint32_t core_id;
    char padding[PAGE_SIZE - sizeof(uint64_t)*12 - sizeof(uint32_t)];
} cpu_writer_phase_timing;
```

**Memory Allocation:**
- Allocate in **host memory** (malloc or std::vector)
- Size: `CPU_NUM_THREADS * sizeof(cpu_timing_data)`
- For multi-phase writers: `4 * sizeof(cpu_writer_phase_timing)`

### Implementation Strategy

#### Phase 1: Infrastructure Setup

**File modifications:** cache_invalidation_testing.cuh, cache_invalidation_testing.cu

1. **Add timing data structures** (as defined above)
2. **Add helper macros** for consistent timing:

```c
// GPU timing helpers
#define GPU_TIMING_START(timing_data, tid) \
    timing_data[tid].start_time = clock64(); \
    timing_data[tid].thread_id = tid;

#define GPU_TIMING_END(timing_data, tid, type, flag) \
    timing_data[tid].end_time = clock64(); \
    timing_data[tid].duration = timing_data[tid].end_time - timing_data[tid].start_time; \
    timing_data[tid].consumer_type = type; \
    timing_data[tid].flag_type = flag;

// CPU timing helpers
#define CPU_TIMING_START(timing_data, idx) \
    auto start_time_##idx = std::chrono::high_resolution_clock::now(); \
    timing_data[idx].core_id = sched_getcpu();

#define CPU_TIMING_END(timing_data, idx, type, flag) \
    auto end_time_##idx = std::chrono::high_resolution_clock::now(); \
    timing_data[idx].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time_##idx.time_since_epoch()).count(); \
    timing_data[idx].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_##idx.time_since_epoch()).count(); \
    timing_data[idx].duration_ns = timing_data[idx].end_ns - timing_data[idx].start_ns; \
    timing_data[idx].consumer_type = type; \
    timing_data[idx].flag_type = flag;
```

3. **Define consumer type constants:**

```c
#define CONSUMER_TYPE_READER_RLX 0
#define CONSUMER_TYPE_READER_ACQ 1
#define CONSUMER_TYPE_WRITER 2
#define CONSUMER_TYPE_DUMMY 3

#define FLAG_TYPE_THREAD 0
#define FLAG_TYPE_BLOCK 1
#define FLAG_TYPE_DEVICE 2
#define FLAG_TYPE_SYSTEM 3
```

#### Phase 2: Modify Main Function (cache_invalidation_testing.cu)

**Changes needed in main():**

1. **Allocate timing arrays:**

```c
// After buffer allocation, add:
gpu_timing_data *gpu_timing;
cpu_timing_data *cpu_timing;

if (reader == CE_GPU || writer == CE_GPU) {
    cudaMalloc(&gpu_timing, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data));
    cudaMemset(gpu_timing, 0, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data));
}

if (reader == CE_CPU || writer == CE_CPU) {
    cpu_timing = (cpu_timing_data*) malloc(CPU_NUM_THREADS * sizeof(cpu_timing_data));
    memset(cpu_timing, 0, CPU_NUM_THREADS * sizeof(cpu_timing_data));
}

// For multi-producer mode, also allocate phase timing
gpu_writer_phase_timing *gpu_phase_timing;
cpu_writer_phase_timing *cpu_phase_timing;

if (multi_producer) {
    cudaMalloc(&gpu_phase_timing, 4 * sizeof(gpu_writer_phase_timing));
    cpu_phase_timing = (cpu_writer_phase_timing*) malloc(4 * sizeof(cpu_writer_phase_timing));
}
```

2. **Pass timing arrays to orchestrator functions:**

```c
// Modify kernel launch to include timing parameter
gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(
    buffer, dummy_buffer, result_g, r_signal, 
    w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal,
    g_spawn_writer,
    gpu_timing  // ADD THIS
);

// Modify CPU thread creation to pass timing
cpu_threads.push_back(std::thread(
    cpu_buffer_reader_writer_propagation_hierarchy, 
    buffer, dummy_buffer, result_c, r_signal,
    w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal,
    &spawn_writer,
    cpu_timing  // ADD THIS
));
```

3. **Copy timing data from device to host:**

```c
// After cudaDeviceSynchronize(), before result printing
if (reader == CE_GPU || writer == CE_GPU) {
    gpu_timing_data *gpu_timing_host = (gpu_timing_data*) malloc(
        GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data)
    );
    cudaMemcpy(gpu_timing_host, gpu_timing, 
               GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data),
               cudaMemcpyDeviceToHost);
    
    // Process and print timing data
    print_gpu_timing_data(gpu_timing_host, GPU_NUM_BLOCKS * GPU_NUM_THREADS);
    
    free(gpu_timing_host);
}

if (reader == CE_CPU || writer == CE_CPU) {
    print_cpu_timing_data(cpu_timing, CPU_NUM_THREADS);
}
```

#### Phase 3: Instrument Consumer Functions

**Pattern for GPU readers:**

```c
template<typename B, typename W, typename R>
__device__ static void __attribute__((optimize("O0"))) 
gpu_buffer_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results, R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing, uint32_t flag_type  // ADD THESE PARAMETERS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Existing pre-cache code...
    
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    // START TIMING AFTER SIGNALING READY
    clock_t timing_start = clock64();
    
    // Wait for writer
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_acquire) == 0) {
        // spin
    }
    
    // Read buffer
    uint result = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_acquire);
    }
    
    // END TIMING AFTER BUFFER READ
    clock_t timing_end = clock64();
    
    // Store timing data
    timing[tid].start_time = timing_start;
    timing[tid].end_time = timing_end;
    timing[tid].duration = timing_end - timing_start;
    timing[tid].consumer_type = CONSUMER_TYPE_READER_ACQ;
    timing[tid].flag_type = flag_type;
    timing[tid].thread_id = tid;
    
    results[tid].data = result;
}
```

**Pattern for GPU multi-phase writers:**

```c
__device__ static void __attribute__((optimize("O0")))
gpu_buffer_writer_propagation_hierarchy(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal, flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal,
    gpu_writer_phase_timing *phase_timing  // ADD THIS
) {
    // Wait for readers
    while(r_signal->flag.load(cuda::memory_order_acquire) != GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {}
    
    // PHASE 1: Thread scope
    phase_timing->phase_start[0] = clock64();
    for (int j = 0; j < BUFFER_SIZE; j++) {
        buffer[j].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    phase_timing->phase_end[0] = clock64();
    phase_timing->phase_duration[0] = phase_timing->phase_end[0] - phase_timing->phase_start[0];
    
    cudaSleep(10000000000);
    
    // PHASE 2: Block scope
    phase_timing->phase_start[1] = clock64();
    for (int j = 0; j < BUFFER_SIZE; j++) {
        buffer[j].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    phase_timing->phase_end[1] = clock64();
    phase_timing->phase_duration[1] = phase_timing->phase_end[1] - phase_timing->phase_start[1];
    
    cudaSleep(10000000000);
    
    // PHASE 3: Device scope
    phase_timing->phase_start[2] = clock64();
    for (int j = 0; j < BUFFER_SIZE; j++) {
        buffer[j].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    phase_timing->phase_end[2] = clock64();
    phase_timing->phase_duration[2] = phase_timing->phase_end[2] - phase_timing->phase_start[2];
    
    cudaSleep(10000000000);
    
    // PHASE 4: System scope
    phase_timing->phase_start[3] = clock64();
    for (int j = 0; j < BUFFER_SIZE; j++) {
        buffer[j].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, P_H_FLAG_STORE_ORDER);
    phase_timing->phase_end[3] = clock64();
    phase_timing->phase_duration[3] = phase_timing->phase_end[3] - phase_timing->phase_start[3];
    
    phase_timing->thread_id = blockIdx.x * blockDim.x + threadIdx.x;
}
```

**Pattern for CPU readers:**

```c
template<typename B, typename R, typename W, typename F>
static void __attribute__((optimize("O0")))
cpu_buffer_reader_propagation_hierarchy_acq(
    B *buffer, bufferElement_na *results, R *r_signal, W *w_signal, F *fallback_signal,
    cpu_timing_data *timing, int thread_idx, uint32_t flag_type  // ADD THESE
) {
    int core_id = sched_getcpu();
    
    // Pre-cache code...
    
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    
    // START TIMING
    auto start = std::chrono::high_resolution_clock::now();
    
    // Wait for writer
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_acquire) == 0) {
        // spin
    }
    
    // Read buffer
    uint result = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_acquire);
    }
    
    // END TIMING
    auto end = std::chrono::high_resolution_clock::now();
    
    // Store timing data
    timing[thread_idx].start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        start.time_since_epoch()).count();
    timing[thread_idx].end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end.time_since_epoch()).count();
    timing[thread_idx].duration_ns = timing[thread_idx].end_ns - timing[thread_idx].start_ns;
    timing[thread_idx].consumer_type = CONSUMER_TYPE_READER_ACQ;
    timing[thread_idx].flag_type = flag_type;
    timing[thread_idx].core_id = core_id;
    
    results[thread_idx].data = result;
}
```

#### Phase 4: Modify Orchestrator Functions

**GPU orchestrator changes:**

```c
__global__ static void __attribute__((optimize("O0")))
gpu_buffer_reader_writer_propagation_hierarchy(
    bufferElement *buffer, bufferElement *w_buffer, bufferElement_na *results,
    flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal, 
    flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal,
    WriterType *spawn_writer,
    gpu_timing_data *timing  // ADD THIS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    
    if (blockId == 0 && threadId == 0 && *spawn_writer != CE_NO_WRITER) {
        // Writer gets no timing (uses phase timing)
        if (*spawn_writer == CE_HET_WRITER) {
            gpu_buffer_writer_propagation_hierarchy_cpu(buffer, r_signal, 
                w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            gpu_buffer_writer_propagation_hierarchy(buffer, r_signal,
                w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else {
        // Readers - pass timing and flag type
        switch(threadId % 8) {
            case 0:
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal, 
                    w_t_signal, fallback_signal, timing, FLAG_TYPE_THREAD);
                break;
            case 1:
                gpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal,
                    w_b_signal, fallback_signal, timing, FLAG_TYPE_BLOCK);
                break;
            // ... etc for all 8 cases
        }
    }
}
```

**CPU orchestrator changes:**

```c
static void __attribute__((optimize("O0")))
cpu_buffer_reader_writer_propagation_hierarchy(
    bufferElement *buffer, bufferElement *w_buffer, bufferElement_na *results,
    flag_d *r_signal, flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal, flag_s *fallback_signal,
    WriterType *spawn_writer,
    cpu_timing_data *timing,  // ADD THIS
    int thread_idx            // ADD THIS
) {
    int core_id = sched_getcpu();
    
    if (core_id == 0 && *spawn_writer != CE_NO_WRITER) {
        // Writer (no per-writer timing, uses phase timing)
        if (*spawn_writer == CE_HET_WRITER) {
            cpu_buffer_writer_propagation_hierarchy_gpu(buffer, r_signal,
                w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        } else {
            cpu_buffer_writer_propagation_hierarchy(buffer, r_signal,
                w_t_signal, w_b_signal, w_d_signal, w_s_signal, fallback_signal);
        }
    } else {
        // Readers
        switch(core_id % 8) {
            case 0:
                cpu_buffer_reader_propagation_hierarchy_rlx(buffer, results, r_signal,
                    w_t_signal, fallback_signal, timing, thread_idx, FLAG_TYPE_THREAD);
                break;
            // ... etc
        }
    }
}
```

#### Phase 5: Output Functions

**Add new functions for printing timing data:**

```c
void print_gpu_timing_data(gpu_timing_data *timing, int count) {
    std::cout << "\n=== GPU TIMING RESULTS ===" << std::endl;
    std::cout << "ThreadID,BlockID,ThreadIdx,ConsumerType,FlagType,StartClock,EndClock,Duration" << std::endl;
    
    for (int i = 0; i < count; i++) {
        if (timing[i].consumer_type == CONSUMER_TYPE_DUMMY) continue;  // Skip dummy workers
        
        int block_id = timing[i].thread_id / GPU_NUM_THREADS;
        int thread_id = timing[i].thread_id % GPU_NUM_THREADS;
        
        const char* consumer_str = 
            (timing[i].consumer_type == CONSUMER_TYPE_READER_RLX) ? "Reader-Rlx" :
            (timing[i].consumer_type == CONSUMER_TYPE_READER_ACQ) ? "Reader-Acq" :
            (timing[i].consumer_type == CONSUMER_TYPE_WRITER) ? "Writer" : "Unknown";
        
        const char* flag_str = 
            (timing[i].flag_type == FLAG_TYPE_THREAD) ? "Thread" :
            (timing[i].flag_type == FLAG_TYPE_BLOCK) ? "Block" :
            (timing[i].flag_type == FLAG_TYPE_DEVICE) ? "Device" :
            (timing[i].flag_type == FLAG_TYPE_SYSTEM) ? "System" : "Unknown";
        
        std::cout << timing[i].thread_id << "," 
                  << block_id << ","
                  << thread_id << ","
                  << consumer_str << ","
                  << flag_str << ","
                  << timing[i].start_time << ","
                  << timing[i].end_time << ","
                  << timing[i].duration << std::endl;
    }
}

void print_cpu_timing_data(cpu_timing_data *timing, int count) {
    std::cout << "\n=== CPU TIMING RESULTS ===" << std::endl;
    std::cout << "CoreID,ConsumerType,FlagType,StartNS,EndNS,DurationNS" << std::endl;
    
    for (int i = 0; i < count; i++) {
        if (timing[i].consumer_type == CONSUMER_TYPE_DUMMY) continue;
        
        const char* consumer_str = 
            (timing[i].consumer_type == CONSUMER_TYPE_READER_RLX) ? "Reader-Rlx" :
            (timing[i].consumer_type == CONSUMER_TYPE_READER_ACQ) ? "Reader-Acq" :
            (timing[i].consumer_type == CONSUMER_TYPE_WRITER) ? "Writer" : "Unknown";
        
        const char* flag_str = 
            (timing[i].flag_type == FLAG_TYPE_THREAD) ? "Thread" :
            (timing[i].flag_type == FLAG_TYPE_BLOCK) ? "Block" :
            (timing[i].flag_type == FLAG_TYPE_DEVICE) ? "Device" :
            (timing[i].flag_type == FLAG_TYPE_SYSTEM) ? "System" : "Unknown";
        
        std::cout << timing[i].core_id << ","
                  << consumer_str << ","
                  << flag_str << ","
                  << timing[i].start_ns << ","
                  << timing[i].end_ns << ","
                  << timing[i].duration_ns << std::endl;
    }
}

void print_gpu_phase_timing(gpu_writer_phase_timing *phase_timing, int writer_count) {
    std::cout << "\n=== GPU WRITER PHASE TIMING ===" << std::endl;
    std::cout << "WriterID,Phase,PhaseName,Start,End,Duration" << std::endl;
    
    const char* phase_names[] = {"Thread", "Block", "Device", "System"};
    
    for (int w = 0; w < writer_count; w++) {
        for (int p = 0; p < 4; p++) {
            std::cout << phase_timing[w].thread_id << ","
                      << p << ","
                      << phase_names[p] << ","
                      << phase_timing[w].phase_start[p] << ","
                      << phase_timing[w].phase_end[p] << ","
                      << phase_timing[w].phase_duration[p] << std::endl;
        }
    }
}

void print_cpu_phase_timing(cpu_writer_phase_timing *phase_timing, int writer_count) {
    std::cout << "\n=== CPU WRITER PHASE TIMING ===" << std::endl;
    std::cout << "CoreID,Phase,PhaseName,StartNS,EndNS,DurationNS" << std::endl;
    
    const char* phase_names[] = {"Thread", "Block", "Device", "System"};
    
    for (int w = 0; w < writer_count; w++) {
        for (int p = 0; p < 4; p++) {
            std::cout << phase_timing[w].core_id << ","
                      << p << ","
                      << phase_names[p] << ","
                      << phase_timing[w].phase_start_ns[p] << ","
                      << phase_timing[w].phase_end_ns[p] << ","
                      << phase_timing[w].phase_duration_ns[p] << std::endl;
        }
    }
}
```

#### Phase 6: Output Format and Analysis

**CSV Format for Easy Parsing:**

Output should be CSV-formatted for easy import into analysis tools (Python pandas, Excel, etc.).

**GPU Timing Output Example:**
```
=== GPU TIMING RESULTS ===
ThreadID,BlockID,ThreadIdx,ConsumerType,FlagType,StartClock,EndClock,Duration
0,0,0,Writer,Thread,123456789,123556789,100000
1,0,1,Reader-Rlx,Thread,123457000,123558000,101000
2,0,2,Reader-Rlx,Block,123457000,123559000,102000
...
```

**CPU Timing Output Example:**
```
=== CPU TIMING RESULTS ===
CoreID,ConsumerType,FlagType,StartNS,EndNS,DurationNS
32,Reader-Rlx,Thread,1703419234567890123,1703419234567990123,100000
33,Reader-Rlx,Block,1703419234567891000,1703419234567992000,101000
...
```

**Analysis Scripts to Create:**

1. **timing_analysis.py** - Parse CSV output and generate:
   - Histograms of propagation delays
   - Per-scope comparison (thread vs block vs device vs system)
   - Acquire vs relaxed comparison
   - Writer phase duration analysis
   - Statistical summary (mean, median, std dev, min, max)

2. **plot_timing.py** - Visualization:
   - Timeline plots showing when each reader observed data
   - Heatmaps showing propagation patterns across cores/threads
   - CDF plots for propagation delays

### Testing Strategy

1. **Unit test each consumer function** independently
2. **Verify timing data is collected** without affecting results
3. **Compare timing overhead** (with vs without timing instrumentation)
4. **Validate CSV output format** can be parsed correctly
5. **Run full test suite** across all memory allocators and configurations

### Performance Considerations

**Timing Overhead:**
- `clock64()`: ~20 cycles on modern GPUs
- `std::chrono::high_resolution_clock::now()`: ~50-100 ns on modern CPUs
- Storing timing data: 1 memory write per field

**Total overhead per consumer:**
- GPU: ~100 cycles + 7 memory writes
- CPU: ~200 ns + 7 memory writes

**Impact:** Minimal (<1% for typical buffer sizes and iteration counts)

**Memory Overhead:**
- GPU: 4KB per thread (page-aligned)
- CPU: 4KB per thread
- Total: ~2MB for GPU (512 threads) + ~128KB for CPU (32 threads)

### Implementation Checklist

- [ ] Phase 1: Add data structures and helper macros
- [ ] Phase 2: Modify main() to allocate and manage timing arrays
- [ ] Phase 3: Instrument all active reader functions (GPU + CPU)
- [ ] Phase 3: Instrument all active writer functions (GPU + CPU)
- [ ] Phase 3: Instrument multi-reader functions
- [ ] Phase 3: Instrument multi-writer functions
- [ ] Phase 4: Modify GPU orchestrator functions
- [ ] Phase 4: Modify CPU orchestrator functions
- [ ] Phase 5: Implement print_gpu_timing_data()
- [ ] Phase 5: Implement print_cpu_timing_data()
- [ ] Phase 5: Implement print_gpu_phase_timing()
- [ ] Phase 5: Implement print_cpu_phase_timing()
- [ ] Phase 6: Create timing_analysis.py
- [ ] Phase 6: Create plot_timing.py
- [ ] Test: Verify timing with GPU-only configuration
- [ ] Test: Verify timing with CPU-only configuration
- [ ] Test: Verify timing with heterogeneous configuration
- [ ] Test: Verify timing with multi-producer mode
- [ ] Test: Compare results with/without timing (validate no interference)
- [ ] Documentation: Update REFERENCE.md with timing output format
- [ ] Documentation: Add timing analysis guide to README.md

### Estimated Implementation Time

- Phase 1-2: 2 hours (infrastructure)
- Phase 3: 4 hours (instrument all consumer functions)
- Phase 4: 2 hours (modify orchestrators)
- Phase 5: 2 hours (output functions)
- Phase 6: 3 hours (analysis scripts)
- Testing: 3 hours
- Documentation: 1 hour

**Total:** ~17 hours

### Alternative Approaches Considered

**Option A: Use CUDA Events Instead of clock64()**
- Pros: More accurate, less overhead
- Cons: Can't time individual threads within kernel, only kernel-level
- Decision: Use clock64() for per-thread granularity

**Option B: Separate Timing Kernels**
- Pros: No modification to existing consumer functions
- Cons: Timing would include kernel launch overhead, less accurate
- Decision: Inline timing within consumer functions

**Option C: Use CUDA Profiler (nvprof/Nsight)**
- Pros: No code changes, powerful analysis
- Cons: Can't distinguish between different consumer types within same kernel
- Decision: Custom timing for fine-grained analysis, use profiler for validation

---

## Summary

### Priorities

1. **HIGH**: Remove redundant functions (reduces confusion, ~450 lines)
2. **HIGH**: Fix bufferElement_d type mismatch (bug)
3. **MEDIUM**: Clean up commented code (improves readability)
4. **MEDIUM**: Implement timing infrastructure (enables measurement)
5. **LOW**: Clean up get_gpu_properties comments (minor cleanup)

### Next Steps

1. Create cleanup branch: `git checkout -b cleanup-redundant-code`
2. Remove all redundant functions (use list above)
3. Fix bufferElement_d type bug
4. Clean commented code
5. Test that everything still compiles and runs
6. Merge cleanup branch
7. Create timing branch: `git checkout -b feature-timing-instrumentation`
8. Implement timing following the plan above
9. Test thoroughly
10. Merge timing branch

### Questions for Review

1. Should we keep get_gpu_properties() commented output or remove entirely?
2. Do we want timing output to stdout or to separate files?
3. Should timing be always-on or behind a compile flag (`-DENABLE_TIMING`)?
4. What analysis metrics are most important for your research?

# Thread Organization Analysis Report

**Date:** December 28, 2025  
**Analysis Target:** CUDA Cache Invalidation Penalty Testing Framework

---

## Executive Summary

This report documents the current thread organization, identifies gaps between intent and implementation, and proposes improvements for specifying thread allocation patterns. The framework tests cache coherence propagation across CPU-GPU heterogeneous systems using various memory ordering semantics and CUDA thread scopes.

**Key Finding:** The current implementation has extensive hard-coded `bid == 99` and `bid == 6` conditions in GPU orchestrators, effectively disabling most actual test readers and replacing them with dummy threads. This significantly limits the experimental coverage compared to the stated intent.

---

## 1. Configuration Space Analysis

### 1.1 Compile-Time Flags

| Category | Flags | Values | Count |
|----------|-------|--------|-------|
| **Memory Ordering (Producer)** | `P_H_FLAG_STORE_ORDER_REL` / `P_H_FLAG_STORE_ORDER_RLX` | Release / Relaxed | 2 |
| **Memory Ordering (Consumer)** | `C_H_FLAG_LOAD_ORDER_ACQ` / `C_H_FLAG_LOAD_ORDER_RLX` | Acquire / Relaxed | 2 |
| **Thread Scope** | `CUDA_THREAD_SCOPE_*` | THREAD, BLOCK, DEVICE, SYSTEM | 4 |
| **Data Size** | `DATA_SIZE_*` | 8, 16, 32, 64 bits | 4 |
| **Other** | `CONSUMERS_CACHE` | Defined / Undefined | 2 |

**Total Compile-Time Combinations:** 4 (ordering combos) × 4 (scopes) × 4 (data sizes) = **64 variants**

The Makefile currently builds: `acq_rel`, `acq_rlx`, `rlx_rel`, `rlx_rlx` × 3 scopes × 4 sizes = **48 executables** (SYSTEM scope excluded)

### 1.2 Runtime Flags

| Flag | Values | Description |
|------|--------|-------------|
| `-m` | `malloc`, `cuda_malloc`, `numa_host`, `numa_device`, `dram`, `um` | Memory allocator selection (6 options) |
| `-r` | `gpu`, `cpu` | Reader device type (2 options) |
| `-w` | `gpu`, `cpu` | Writer device type (2 options) |
| `-p` | (present/absent) | Multi-producer mode toggle (2 options) |

**Total Runtime Combinations:** 6 × 2 × 2 × 2 = **96 configurations**

**Total Experimental Space:** 64 × 96 = **6,144 possible test configurations**

---

## 2. Execution Modes

### 2.1 Single-Writer Mode (`-p` absent)

**Purpose:** Test propagation of a single writer's updates across the scope hierarchy  
**Writer Behavior:** Phased writes with increasing buffer values and scope progression

#### Writer Progression Pattern:
```
buffer[i] = 10  → set flag_thread   → cudaSleep(10B cycles)
buffer[i] = 20  → set flag_block    → cudaSleep(10B cycles)
buffer[i] = 30  → set flag_device   → cudaSleep(10B cycles)
buffer[i] = 40  → set flag_system   → cudaSleep(10B cycles)
buffer[i] = 50  → set fallback = 4  → cudaSleep(10B cycles)
```

#### Expected Test: 
Each reader waits on a specific flag scope and ordering, observing what buffer value is visible when their flag propagates.

### 2.2 Multi-Producer Mode (`-p` present)

**Purpose:** Test concurrent visibility with 4 simultaneous writers, each owning a scope level  
**Writers:** Thread/Block/Device/System writers operate concurrently on separate buffers

#### Writer Pattern (per scope):
```
Writer_T: buffer_t[i] = 10 → flag_t = 1 → sleep → buffer_t[i] = 1
Writer_B: buffer_b[i] = 20 → flag_b = 1 → sleep → buffer_b[i] = 2
Writer_D: buffer_d[i] = 30 → flag_d = 1 → sleep → buffer_d[i] = 3
Writer_S: buffer_s[i] = 40 → flag_s = 1 → sleep → buffer_s[i] = 4
```

**Buffers:** 4 separate typed buffers (`bufferElement_t/b/d/s`) with matching atomic scopes

---

## 3. Thread Organization: Current Implementation

### 3.1 GPU Thread Organization

**Configuration:** 8 blocks × 64 threads = **512 total threads**

#### Single-Writer Mode (`gpu_buffer_reader_writer_propagation_hierarchy`)

**Block 0:**
- Thread 0: Writer (if `spawn_writer != CE_NO_WRITER`)
- Threads 1-63:
  - `tid % 8 == 0`: Read flag_thread
  - `tid % 8 == 1`: Read flag_block
  - `tid % 8 == 2`: Read flag_device
  - `tid % 8 == 3`: Read flag_system
  - `tid % 8 == 4-7`: Dummy readers (w_buffer)

**Blocks 1-7:**
- Thread 0: Dummy writer (w_buffer)
- Threads 1-31:
  - **CRITICAL ISSUE:** Hard-coded `bid == 6` and `bid == 99` filters
  - `tid % 8 == 0`: Actual reader **ONLY if `bid == 6`**, else dummy
  - `tid % 8 == 1-3`: Actual readers **ONLY if `bid == 99`** (never executed!)
  - `tid % 8 == 4-7`: Actual readers **ONLY if `bid == 99`** (never executed!)
- Threads 32-63: All dummy readers

**Effective Distribution:**
- Block 0: 1 writer + 7 actual readers (scope T) + 7 readers (B/D/S) + 48 dummy readers = 63 threads
- Block 6: 1 dummy writer + 3 actual readers (scope T only) + 60 dummy readers = 64 threads
- Blocks 1-5, 7: 64 dummy threads each = 384 threads

**Total Actual Readers:** ~10-15 threads out of 512 (~2-3%)

#### Multi-Producer Mode (`gpu_buffer_reader_multi_writer_propagation_hierarchy`)

**Blocks 0-3:**
- Thread 0: Writer (one scope per block)
  - Block 0: Thread-scope writer
  - Block 1: Block-scope writer
  - Block 2: Device-scope writer
  - Block 3: System-scope writer
- Threads 1-63: Readers based on `global_tid % 8`

**Blocks 4-7:**
- All threads: Readers based on `global_tid % 8`
- **CRITICAL ISSUE:** `bid == 5` and `bid == 99` filters again!
  - `global_tid % 8 == 0`: Actual reader **ONLY if `bid == 5`**
  - `global_tid % 8 == 1-7`: Actual readers **ONLY if `bid == 99`** (never!)

**Effective Distribution:**
- Blocks 0-3: 4 writers + 252 readers (but filtered by `% 8` pattern)
- Block 5: 64 readers (only `% 8 == 0` are actual, rest dummy)
- Other blocks: Nearly all dummy

### 3.2 CPU Thread Organization

**Configuration:** 32 threads pinned to cores 32-63

#### Single-Writer Mode (`cpu_buffer_reader_writer_propagation_hierarchy`)

Thread assignment based on `core_id % 8` and `core_id % 32`:

- **Core 32 (core_id % 32 == 0):** Writer
- **Cores 33-63:** Readers
  - `core_id % 8 == 0`: Read flag_thread
  - `core_id % 8 == 1`: Read flag_block
  - `core_id % 8 == 2`: Read flag_device
  - `core_id % 8 == 3`: Read flag_system
  - `core_id % 8 == 4-7`: Conditional based on `NO_ACQ` define
    - If `NO_ACQ` defined: Dummy readers
    - Else: Read flag_thread/block/device/system (acquire versions)

**Distribution:**
- 1 writer
- 31 readers: 4 scopes × 2 ordering patterns (relaxed on cores %8==0-3, acquire on %8==4-7)
- **Pattern:** Repeating 8-thread groups create 4 sets across 32 threads

#### Multi-Producer Mode (`cpu_buffer_reader_multi_writer_propagation_hierarchy`)

Thread assignment based on `core_id / 8` and `core_id % 8`:

**Writers (core_id % 8 == 0):**
- Core 32: Thread-scope writer
- Core 40: Block-scope writer
- Core 48: Device-scope writer
- Core 56: System-scope writer

**Readers (core_id % 8 != 0):**
- `core_id % 8 == 0`: Thread-scope reader (but this is writer!)
- `core_id % 8 == 1`: Block-scope reader
- `core_id % 8 == 2`: Device-scope reader
- `core_id % 8 == 3`: System-scope reader
- `core_id % 8 == 4-7`: Dummy readers

**Distribution:**
- 4 writers
- 12 actual readers (scopes B/D/S, 4 each)
- 16 dummy readers

---

## 4. Dummy Thread Analysis

### 4.1 Purpose of Dummy Threads

Dummy threads operate on an **independent buffer** (`dummy_buffer` or `w_buffer`) to:
1. Generate memory subsystem traffic
2. Create cache pressure and contention
3. Simulate realistic concurrent workload conditions
4. Test propagation delays under load

### 4.2 Dummy Thread Functions

**GPU:**
- `gpu_dummy_writer_worker_propagation`: Writes to w_buffer, signals readiness, loops for NUM_ITERATIONS
- `gpu_dummy_reader_worker_propagation`: Reads from w_buffer, signals readiness, loops for NUM_ITERATIONS

**CPU:**
- `cpu_dummy_reader_worker_propagation`: Reads from w_buffer, signals readiness, loops for NUM_ITERATIONS

**Characteristics:**
- All use relaxed memory ordering
- Loop for `NUM_ITERATIONS = 10,000` × `BUFFER_SIZE = 512` = 5.12M operations per dummy thread
- GPU dummy writer includes 4 long sleep cycles (10B cycles) during execution

### 4.3 Current Dummy Thread Distribution

#### GPU Single-Writer Mode:
- Block 0: ~48 dummy readers (75%)
- Block 6: 1 dummy writer + ~60 dummy readers (95%)
- Blocks 1-5, 7: 1 dummy writer + 63 dummy readers per block (100%)
- **Total: ~470 dummy threads out of 512 (92%)**

#### GPU Multi-Producer Mode:
- Blocks 0-3: Filtered by `global_tid % 8`, approximately 7/8 dummy
- Blocks 4-7: Nearly all dummy due to `bid == 99` filter
- **Total: ~420 dummy threads out of 508 (83%)**

#### CPU Single-Writer Mode:
- 0 or 15-16 dummy threads out of 31 (depends on `NO_ACQ` define)
- **Total: 0-52% dummy**

#### CPU Multi-Producer Mode:
- 16 dummy threads out of 28 (57%)

### 4.4 Dummy Thread Balance Assessment

**Issues Identified:**

1. **GPU Imbalance:** 83-92% dummy threads overwhelming actual test threads
2. **Block Distribution:** Dummy operations not evenly distributed across CTAs
3. **Logical Unit Isolation:** No clear separation between "test blocks" and "dummy blocks"
4. **Warp Mixing:** In Block 0, mix of 8-thread patterns within warps may not create intended interference

**Missing Patterns:**
- No pure "observation blocks" where all threads are actual readers
- No controlled mixing within warps (e.g., 1 acquire + 31 relaxed in a warp)
- No CTA-level scope experiments (e.g., one CTA with device-acquire, rest with thread-relaxed)

---

## 5. Gap Analysis: Intent vs. Implementation

### 5.1 Desired Experimental Patterns (From User Intent)

You specified wanting patterns like:

1. **CTA A:** 1 device-scoped acquire, 63 thread-scoped relaxed
2. **CTA B:** 1 device-scoped relaxed, 63 thread-scoped relaxed
3. **CTA C:** 1 system-scoped relaxed, 63 thread-scoped relaxed
4. **CTA D:** 1 system-scoped acquire, 63 device-scoped relaxed

### 5.2 Current Implementation Gaps

| Desired Feature | Current Status | Gap |
|----------------|----------------|-----|
| Per-CTA scope mixing | ❌ Not implemented | Cannot specify different scope patterns per CTA |
| Per-CTA ordering mixing | ❌ Limited | `C_H_FLAG_LOAD_ORDER` is global compile-time flag |
| Singular acquire in CTA | ❌ Not supported | All threads in pattern use same ordering |
| Cross-scope comparison | ⚠️ Partial | `% 8` pattern gives different scopes but same ordering per thread |
| Repeatable warp patterns | ✅ Implemented | `% 8`, `% 4`, `% 32` create patterns |
| Dummy/test separation | ⚠️ Poor | Hard-coded `bid` filters, no systematic allocation |
| Independent dummy buffer | ✅ Implemented | `dummy_buffer` / `w_buffer` exists |
| Configurable thread roles | ❌ Not supported | All thread assignment hard-coded in orchestrators |

### 5.3 Missing Combinations

The current `% 8` pattern in GPU code creates:
- 4 different scopes (Thread, Block, Device, System)
- But **NOT** different orderings per thread within a CTA

To achieve "1 acquire + 63 relaxed in a CTA", you need:
- **Per-thread ordering control** (currently ordering is compile-time global)
- Or **conditional ordering in code** based on thread ID

**Example Missing Pattern:**
```cuda
// Desired: Block 0 has 1 device-acquire, rest are thread-relaxed
if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Device scope, acquire ordering
    while (w_d_signal->flag.load(cuda::memory_order_acquire) == 0);
} else if (blockIdx.x == 0) {
    // Thread scope, relaxed ordering
    while (w_t_signal->flag.load(cuda::memory_order_relaxed) == 0);
}
```

**Current Implementation:** All threads in a CTA use the same `C_H_FLAG_LOAD_ORDER`.

---

## 6. Propagation Hierarchy Analysis

### 6.1 Writer Phasing (Single-Writer Mode)

The single writer sets flags in hierarchical order:

| Phase | Buffer Value | Flag Set | Delay | Expected Observation |
|-------|-------------|----------|-------|---------------------|
| 1 | 10 | `flag_thread = 1` | 10B cycles | Thread-scope readers see 10 first |
| 2 | 20 | `flag_block = 1` | 10B cycles | Block-scope readers see 20 |
| 3 | 30 | `flag_device = 1` | 10B cycles | Device-scope readers see 30 |
| 4 | 40 | `flag_system = 1` | 10B cycles | System-scope readers see 40 |
| 5 | 50 | `fallback = 4` | - | Timeout mechanism |

**Intent:** Observe how scope hierarchy affects visibility timing.

**Current Issue:** With only ~10 actual readers, statistical significance is low.

### 6.2 Multi-Writer Mode Concurrency

Four writers operate simultaneously, each on a different buffer/flag:

| Writer | Buffer | Value 1 | Value 2 | Flag Scope |
|--------|--------|---------|---------|-----------|
| Thread | buffer_t | 10 | 1 | thread |
| Block | buffer_b | 20 | 2 | block |
| Device | buffer_d | 30 | 3 | device |
| System | buffer_s | 40 | 4 | system |

**Intent:** Test concurrent propagation at different scope levels.

**Current Issue:** Readers are distributed across scopes by `% 8`, but no designed cross-observation (e.g., thread-scope reader observing device-scope writer).

---

## 7. Proposed Improvements

### 7.1 Thread Allocation Specification System

**Design Goals:**
1. Human-readable configuration format
2. Per-CTA thread role assignment
3. Per-thread scope and ordering control
4. Separate test and dummy thread allocation
5. Runtime-configurable patterns

**Proposed Format: YAML Configuration**

```yaml
# Example: thread_pattern.yaml
thread_pattern:
  gpu:
    num_blocks: 8
    threads_per_block: 64
    
    block_0:
      thread_0:
        role: writer
        scope: device
        ordering: release
      threads_1_7:
        role: reader
        scope: thread
        ordering: relaxed
        flags: [thread]
      thread_8:
        role: reader
        scope: device
        ordering: acquire
        flags: [device]
      threads_9_63:
        role: reader
        scope: thread
        ordering: relaxed
        flags: [thread]
    
    block_1:
      pattern: all_dummy
      dummy_type: reader
    
    block_2:
      thread_0:
        role: reader
        scope: system
        ordering: acquire
        flags: [system]
      threads_1_63:
        role: reader
        scope: device
        ordering: relaxed
        flags: [device]
    
    blocks_3_7:
      pattern: mixed
      ratio: "50% dummy, 50% test"
      test_threads:
        scope: [thread, block, device, system]  # cyclic distribution
        ordering: relaxed
  
  cpu:
    num_threads: 32
    affinity: [32-63]
    
    thread_0:
      role: writer
      scope: system
      ordering: release
    
    threads_1_31:
      pattern: repeating_8
      core_mod_8:
        0-3:
          role: reader
          scopes: [thread, block, device, system]  # cyclic
          ordering: relaxed
        4-7:
          role: reader
          scopes: [thread, block, device, system]  # cyclic
          ordering: acquire
```

### 7.2 Implementation Strategy

**Phase 1: Parameterize Existing Patterns**

Add runtime parameters to control:
- Which blocks have actual readers vs. dummy
- Thread-to-scope mapping
- Dummy thread percentage

```cpp
struct ThreadPattern {
    int actual_reader_blocks[8];  // Block IDs with actual readers
    int dummy_block_ratio;        // Percentage of threads that are dummy in actual blocks
    int scope_assignment[64];     // Per-thread scope assignment (0=T, 1=B, 2=D, 3=S)
    int ordering_assignment[64];  // Per-thread ordering (0=relaxed, 1=acquire)
};
```

**Phase 2: Code Generation Approach**

Generate orchestrator functions from configuration:

```python
# func_decl.py extension
def generate_orchestrator(pattern_config):
    """Generate custom orchestrator based on pattern config."""
    code = []
    for block_id in range(pattern_config['num_blocks']):
        block_cfg = pattern_config[f'block_{block_id}']
        for thread_id in range(pattern_config['threads_per_block']):
            thread_cfg = block_cfg.get_thread_config(thread_id)
            if thread_cfg['role'] == 'writer':
                code.append(generate_writer_call(thread_cfg))
            elif thread_cfg['role'] == 'reader':
                code.append(generate_reader_call(thread_cfg))
            elif thread_cfg['role'] == 'dummy':
                code.append(generate_dummy_call(thread_cfg))
    return "\n".join(code)
```

**Phase 3: Template-Based Orchestrators**

Use C++ templates to support runtime pattern specification:

```cuda
template <int BlockId, int ThreadId>
__device__ void dispatch_thread_role(
    ThreadPatternConfig* config,
    /* ... buffers and signals ... */
) {
    ThreadRole role = config->get_role(BlockId, ThreadId);
    ThreadScope scope = config->get_scope(BlockId, ThreadId);
    MemoryOrder ordering = config->get_ordering(BlockId, ThreadId);
    
    if (role == WRITER) {
        dispatch_writer<scope, ordering>(...);
    } else if (role == READER) {
        dispatch_reader<scope, ordering>(...);
    } else if (role == DUMMY_READER) {
        gpu_dummy_reader_worker_propagation(...);
    } else if (role == DUMMY_WRITER) {
        gpu_dummy_writer_worker_propagation(...);
    }
}
```

### 7.3 Immediate Fixes

**Quick Win 1: Remove Hard-Coded Block Filters**

Change:
```cuda
if (bid == 99)
    gpu_buffer_reader_propagation_hierarchy(...);
else 
    gpu_dummy_reader_worker_propagation(...);
```

To:
```cuda
// Enable all blocks for actual readers
gpu_buffer_reader_propagation_hierarchy(...);
```

**Quick Win 2: Add Runtime Block Selection**

```cuda
__global__ void orchestrator(
    /* ... */,
    int* actual_reader_blocks,  // Array of block IDs that should run actual readers
    int num_actual_blocks
) {
    bool is_actual_block = false;
    for (int i = 0; i < num_actual_blocks; i++) {
        if (blockIdx.x == actual_reader_blocks[i]) {
            is_actual_block = true;
            break;
        }
    }
    
    if (is_actual_block) {
        // Actual test logic
    } else {
        // Dummy logic
    }
}
```

**Quick Win 3: Add Command-Line Pattern Selection**

```bash
./executable -m malloc -r gpu -w gpu \
    --gpu-pattern "block0=actual,block1-7=dummy" \
    --cpu-pattern "core32=writer,core33-63=readers"
```

---

## 8. Specific Pattern Examples

### 8.1 Pattern A: Isolated Acquire in CTA

**Goal:** Test if single acquire in CTA affects other relaxed loads in same CTA

**Configuration:**
- Block 0, Thread 0: Writer (device scope, release)
- Block 1, Thread 0: Device-scope acquire reader
- Block 1, Threads 1-63: Thread-scope relaxed readers
- Blocks 2-7: All dummy

**Expected Behavior:** 
- Observe if Block 1's Thread 0 acquire causes faster propagation to Threads 1-63
- Compare with Block 2 (all relaxed) to isolate effect

### 8.2 Pattern B: Scope Hierarchy Comparison

**Goal:** Compare propagation speed across scopes at same ordering

**Configuration:**
- Block 0, Thread 0: Writer
- Block 1: All threads thread-scope relaxed
- Block 2: All threads block-scope relaxed
- Block 3: All threads device-scope relaxed
- Block 4: All threads system-scope relaxed
- Blocks 5-7: Dummy

**Expected Behavior:**
- Measure which block observes buffer updates first
- Validate scope hierarchy semantics

### 8.3 Pattern C: Warp-Level Mixing

**Goal:** Test acquire/relaxed effects within a warp

**Configuration:**
- Block 0, Thread 0: Writer
- Block 1, Warp 0:
  - Thread 0: Device-scope acquire
  - Threads 1-31: Thread-scope relaxed
- Block 1, Warp 1:
  - All threads: Thread-scope relaxed (control group)
- Rest: Dummy

**Expected Behavior:**
- Observe if acquire in Thread 0 affects Threads 1-31 in same warp
- Compare with Warp 1 to isolate warp-level effects

---

## 9. Recommendations

### 9.1 Short-Term (Immediate Actions)

1. **Fix Hard-Coded Filters:** Remove `bid == 99` and `bid == 6` conditionals to enable all blocks
2. **Document Current Patterns:** Add comments to orchestrator functions explaining `% 8` patterns
3. **Add Runtime Flags:** Implement `--actual-blocks` flag to select which blocks run actual tests
4. **Reduce Dummy Ratio:** Target 30-50% dummy threads instead of 92%

### 9.2 Medium-Term (1-2 Weeks)

1. **Implement Pattern Configuration:** Design and implement YAML/JSON config format
2. **Add Code Generator:** Extend `func_decl.py` to generate orchestrators from configs
3. **Separate Ordering Control:** Enable per-thread ordering specification (not just compile-time global)
4. **Add Verification:** Implement sanity checks to ensure thread counts match expectations

### 9.3 Long-Term (1-2 Months)

1. **Full Pattern Library:** Create pre-defined pattern templates for common experiments
2. **Auto-Validation:** Add runtime validation that pattern constraints are met
3. **Visualization Tools:** Generate diagrams showing thread allocation for each pattern
4. **Statistical Analysis:** Integrate result analysis to auto-detect significance of scope/ordering effects

---

## 10. Conclusion

The current implementation provides a solid foundation for cache coherence testing but has significant gaps in thread organization flexibility:

**Strengths:**
- ✅ Well-defined single-writer and multi-producer modes
- ✅ Phased writer progression across scope hierarchy
- ✅ Independent dummy buffer for traffic generation
- ✅ Repeatable patterns via modulo operations

**Critical Issues:**
- ❌ Hard-coded block filters limit actual test coverage to ~2-3% of GPU threads
- ❌ Cannot specify per-CTA scope/ordering combinations
- ❌ Excessive dummy thread ratio (92%) reduces statistical power
- ❌ No support for "1 acquire + 63 relaxed" patterns within a CTA

**Priority Actions:**
1. Remove `bid == 99` filters to enable all blocks
2. Implement runtime block selection flags
3. Design pattern configuration system
4. Reduce dummy thread ratio to 30-50%

**Impact:** These improvements will enable the originally intended experimental patterns and provide comprehensive coverage of scope × ordering interaction effects across the cache coherence hierarchy.

---

## Appendix A: Thread Assignment Tables

### A.1 GPU Single-Writer Mode (Current)

| Block | Thread 0 | Threads 1-7 (% 8) | Threads 8-31 | Threads 32-63 | Actual Test | Dummy |
|-------|----------|------------------|--------------|---------------|-------------|-------|
| 0 | Writer | T/B/D/S readers + 4 dummy | Conditional (blocked) | All dummy | 8-15 | 48-56 |
| 1-5 | Dummy W | All dummy (blocked) | All dummy | All dummy | 0 | 64 |
| 6 | Dummy W | 3 actual (only % 8 == 0) | All dummy | All dummy | 3 | 61 |
| 7 | Dummy W | All dummy (blocked) | All dummy | All dummy | 0 | 64 |

### A.2 CPU Single-Writer Mode (Current)

| Core ID | Role | Scope | Ordering | Notes |
|---------|------|-------|----------|-------|
| 32 (% 32 == 0) | Writer | System | Release | Single writer |
| 33 (% 8 == 1) | Reader | Block | Relaxed | Repeating pattern |
| 34 (% 8 == 2) | Reader | Device | Relaxed | |
| 35 (% 8 == 3) | Reader | System | Relaxed | |
| 36 (% 8 == 4) | Reader/Dummy | Thread | Acquire/NA | If NO_ACQ: dummy |
| 37 (% 8 == 5) | Reader/Dummy | Block | Acquire/NA | If NO_ACQ: dummy |
| ... | (pattern repeats) | ... | ... | 4 groups of 8 |

---

**End of Report**
