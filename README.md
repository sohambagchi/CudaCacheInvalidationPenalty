# CUDA Cache Invalidation Penalty Testing

A heterogeneous CPU-GPU test framework for measuring cache coherence propagation delays across different CUDA thread scopes and memory ordering semantics.

## Purpose

This framework tests cache invalidation penalties when data written by one processor (CPU or GPU) needs to be visible to another processor. It uses a **pattern-based configuration system** where test scenarios are defined in YAML files, allowing flexible specification of:
- Thread roles (writer, reader, dummy threads)
- Memory ordering semantics (acquire, release, relaxed)
- CUDA thread scopes (thread, block, device, system)
- Per-thread behavior and synchronization patterns

## Quick Start

### Build

```bash
make all    # Build all data size variants
make clean  # Remove build artifacts
```

This creates 4 executables in `output/`:
- `cache_invalidation_testing_DATA_SIZE_8.out`
- `cache_invalidation_testing_DATA_SIZE_16.out`
- `cache_invalidation_testing_DATA_SIZE_32.out`
- `cache_invalidation_testing_DATA_SIZE_64.out`

### Run

```bash
# List available patterns in a config file
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -F configs/isolated_acquire.yaml

# Run specific pattern
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -P isolated_acquire_per_cta \
    -F configs/isolated_acquire.yaml \
    -m um

# Run multi-writer pattern
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -P multi_writer_test \
    -F configs/test_multi_writer.yaml \
    -m dram
```

### Command-Line Options

**Required:**
- `-P <pattern_name>` - Name of pattern to execute (omit to list patterns)
- `-F <yaml_file>` - Path to pattern configuration file

**Optional:**
- `-m <allocator>` - Memory allocator type (default: malloc)
- `malloc` - System malloc (default)
- `dram` - CUDA pinned host memory (`cudaMallocHost`)
- `um` - CUDA unified memory (`cudaMallocManaged`)
- `numa_host` - NUMA allocation on CPU node
- `numa_device` - NUMA allocation on GPU node
- `cuda_malloc` - GPU device memory (GPU-only)

## Pattern-Based Configuration

### YAML Pattern Format

Test configurations are defined in YAML files in the `configs/` directory. Each pattern specifies per-thread behavior:

```yaml
patterns:
  - name: "pattern_name"
    description: "What this pattern tests"
    multi_writer: false  # Optional, enables multi-writer mode
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_31: {role: reader, ordering: acquire, watch_flag: device}
          threads_32_63: {role: dummy_reader}
        
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      threads:
        all_threads: {role: inactive}
```

### Thread Roles

- **writer** - Writes to buffer and sets synchronization flag
- **reader** - Waits for flag, then reads buffer
- **dummy_writer** - Background traffic (writes to separate buffer)
- **dummy_reader** - Background traffic (reads from separate buffer)
- **inactive** - Thread does nothing

### Memory Ordering

- **release** - Writer uses `memory_order_release`
- **acquire** - Reader uses `memory_order_acquire`
- **relaxed** - Uses `memory_order_relaxed`
- **acq_rel** - Uses `memory_order_acq_rel`

### Thread Scopes

- **thread** - Thread-level atomic scope
- **block** - Block-level atomic scope
- **device** - Device-level atomic scope
- **system** - System-level atomic scope

### Multi-Writer Mode

Enable with `multi_writer: true` in pattern YAML. This mode:
- Requires exactly 4 writers (one per scope: thread, block, device, system)
- Allocates 4 scope-specific buffers
- Routes writers to their scope-specific buffer
- Routes readers to buffer matching their `watch_flag`

Example:
```yaml
patterns:
  - name: multi_writer_test
    multi_writer: true
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: thread, ordering: release}
          thread_1: {role: writer, scope: block, ordering: release}
          thread_2: {role: writer, scope: device, ordering: release}
          thread_3: {role: writer, scope: system, ordering: release}
          threads_4_7: {role: reader, ordering: acquire, watch_flag: thread}
          threads_8_11: {role: reader, ordering: acquire, watch_flag: block}
          # ... more readers
```

## Key Concepts

### Pattern Dispatch System

The framework uses a pattern-based dispatch system where:
1. Patterns are defined in YAML configuration files
2. Pattern configuration is loaded at runtime
3. Each thread (GPU/CPU) is assigned a specific role and behavior
4. Dispatch functions route threads to appropriate consumer functions
5. Results are collected and displayed

### Synchronization Pattern

1. **Readers signal ready:** Each reader increments a readiness counter
2. **Writer waits:** Writer spins until all readers are ready
3. **Writer updates:** Writer writes to buffer
4. **Writer signals:** Writer sets completion flag with specified memory ordering
5. **Readers wait:** Readers spin on flag with specified memory ordering
6. **Readers consume:** Readers read buffer and store results
7. **Timing:** Framework can measure propagation delays

### Scope Hierarchy

The framework tests visibility across CUDA's memory scope hierarchy:
1. **Thread scope** - Visibility within a single thread
2. **Block scope** - Visibility within a thread block
3. **Device scope** - Visibility across GPU
4. **System scope** - Visibility across CPU-GPU system

### Dummy Threads

Dummy threads generate background memory traffic without participating in synchronization:
- Operate on separate `dummy_buffer`
- Use relaxed memory ordering
- Loop for `NUM_ITERATIONS` (10,000)
- Create realistic cache pressure and contention

## File Overview

### Core Files
- [src/cache_invalidation_testing.cu](src/cache_invalidation_testing.cu) - Main entry point and orchestration
- [src/pattern_config.cpp](src/pattern_config.cpp) - YAML parsing and validation
- [include/pattern_dispatch.cuh](include/pattern_dispatch.cuh) - GPU dispatch and consumer functions
- [include/pattern_dispatch_cpu.hpp](include/pattern_dispatch_cpu.hpp) - CPU consumer functions
- [include/pattern_config.hpp](include/pattern_config.hpp) - Pattern configuration structures
- [include/types.hpp](include/types.hpp) - Data structures, enums, and constants

### Configuration Files
- `configs/isolated_acquire.yaml` - Tests for isolated acquire effects
- `Architecture

- **GPU:** 8 blocks × 64 threads = 512 threads
- **CPU:** 32 threads with core affinity
- **Buffer:** 512 elements, each padded to 4KB (page-aligned)
- **Iterations:** 10,000 per test

## Data Types

Build variants support different atomic data sizes:
- `DATA_SIZE_8` - uint8_t
- `DATA_SIZE_16` - uint16_t
- `DATA_SIZE_32` - uint32_t (most common)
- `DATA_SIZE_64` - uint64_t

## Example Patterns

### Basic Release-Acquire Test

```yaml
patterns:
  - name: "basic_test"
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

### Isolated Acquire Test

```yaml
patterns:
  - name: "isolated_acquire"
    description: "Test if single acquire affects other relaxed loads in same CTA"
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        block_1:
          # One acquire + 63 relaxed - does acquire affect relaxed?
          thread_0: {role: reader, ordering: acquire, watch_flag: device}
          threads_1_63: {role: reader, ordering: relaxed, watch_flag: device}
        
        block_2:
          # All relaxed (control group)
          all_threads: {role: reader, ordering: relaxed, watch_flag: device}
```

## Contributing

When creating new patterns:
1. Add YAML file to `configs/`
2. Use descriptive pattern names
3. Include detailed description field
4. Test with different memory allocators
5. Document expected behavior

## Troubleshooting

**Pattern not found:**
```bash
# List all patterns in file
./output/cache_invalidation_testing_DATA_SIZE_32.out -F configs/your_file.yaml
```

**Build errors:**
- Ensure CUDA toolkit is installed and in PATH
- Check architecture target in Makefile (`-arch=sm_87`)
- Verify libnuma is installed: `sudo apt-get install libnuma-dev`

**Validation errors:**
- Check YAML syntax
- Ensure multi-writer patterns have exactly 4 writers (one per scope)
- Verify all readers have valid `watch_flag` values
- `CUDA_THREAD_SCOPE_THREAD` - Thread-level atomic scope
- `CUDA_THREAD_SCOPE_BLOCK` - Block-level atomic scope
- `CUDA_THREAD_SCOPE_DEVICE` - Device-level atomic scope
- `CUDA_THREAD_SCOPE_SYSTEM` - System-level atomic scope

### Data Size
- `DATA_SIZE_8` - 8-bit data elements
- `DATA_SIZE_16` - 16-bit data elements
- `DATA_SIZE_32` - 32-bit data elements (default)
- `DATA_SIZE_64` - 64-bit data elements

### Memory Ordering (Producer/Writer)
- `P_H_FLAG_STORE_ORDER_REL` - Writer uses `memory_order_release` for flag stores
- `P_H_FLAG_STORE_ORDER_RLX` - Writer uses `memory_order_relaxed` for flag stores

### Memory Ordering (Consumer/Reader)
- `C_H_FLAG_LOAD_ORDER_ACQ` - Reader uses `memory_order_acquire` for flag loads
- `C_H_FLAG_LOAD_ORDER_RLX` - Reader uses `memory_order_relaxed` for flag loads

### Other Options
- `CONSUMERS_CACHE` - Enable pre-caching in consumer functions

## Architecture Notes

- GPU: 8 blocks × 64 threads = 512 threads
- CPU: 32 threads with core affinity
- Buffer: 512 elements, each padded to 4KB (page-aligned)
- Iterations: 10,000 per test
