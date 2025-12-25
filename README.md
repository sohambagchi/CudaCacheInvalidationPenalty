# CUDA Cache Invalidation Penalty Testing

A heterogeneous CPU-GPU test framework for measuring cache coherence propagation delays across different CUDA thread scopes and memory ordering semantics.

## Purpose

This codebase tests cache invalidation penalties when data written by one processor (CPU or GPU) needs to be visible to another processor. It specifically measures how memory ordering semantics (`memory_order_release`, `memory_order_acquire`, `memory_order_relaxed`) and CUDA thread scopes (`thread`, `block`, `device`, `system`) affect cache coherence propagation.

## Quick Start

### Build

```bash
make all              # Build all variants
make flag-rel         # Build release-ordered flag variants only
make flag-rlx         # Build relaxed-ordered flag variants only
```

### Run

```bash
# Single executable with runtime flags
./cache_invalidation_testing_rel_CUDA_THREAD_SCOPE_THREAD_DATA_SIZE_32_BUFFER_SAME.out \
    -m malloc \       # Memory allocator
    -r gpu \          # Reader type (gpu/cpu)
    -w cpu \          # Writer type (gpu/cpu)
    -p                # Enable multi-producer mode

# Run all configurations
python3 run_all.py <prefix>
```

### Memory Allocators (`-m`)
- `malloc` - System malloc
- `dram` - CUDA pinned host memory (`cudaMallocHost`)
- `um` - CUDA unified memory (`cudaMallocManaged`)
- `numa_host` - NUMA allocation on CPU node
- `numa_device` - NUMA allocation on GPU node
- `cuda_malloc` - GPU device memory (GPU-only)

### Reader/Writer Types (`-r`, `-w`)
- `gpu` - GPU threads as consumers
- `cpu` - CPU threads as consumers

### Modes
- Normal: Single writer tests propagation across thread scope hierarchy
- Multi-producer (`-p`): Four writers (one per scope level) test concurrent visibility

## Key Concepts

### Propagation Hierarchy
The code tests how quickly updates propagate through CUDA's memory scope hierarchy:
1. **Thread scope** - Visibility within a single thread
2. **Block scope** - Visibility within a thread block
3. **Device scope** - Visibility across GPU
4. **System scope** - Visibility across CPU-GPU system

### Synchronization Pattern
1. Readers spin on a reader signal until all are ready
2. Writer waits for reader signal, then writes data
3. Writer sets writer signal(s) with different memory orders
4. Readers acquire writer signal and re-read buffer
5. Timing measurements capture propagation delays

## File Overview

- `cache_invalidation_testing.cuh` - All function declarations and definitions (2178 lines)
- `cache_invalidation_testing.cu` - Main entry point and runtime logic
- `Makefile` - Build system with compile-time flag matrix
- `run_all.py` - Automated test runner
- `check_distribution.py` - Result analysis utilities
- `func_decl.py` - Code generation helper

## Documentation

See [REFERENCE.md](REFERENCE.md) for detailed technical reference including:
- All consumer function signatures
- Data structure layouts
- Synchronization mechanisms
- Timing insertion points for instrumentation

## Compile-Time Flags

- `CUDA_THREAD_SCOPE_*` - Sets atomic scope
- `DATA_SIZE_*` - Sets data element size (8/16/32/64-bit)
- `P_H_FLAG_STORE_ORDER_REL` - Writer uses release store
- `P_H_FLAG_STORE_ORDER_RLX` - Writer uses relaxed store
- `NO_ACQ` - Disable acquire loads in readers
- `CONSUMERS_CACHE` - Enable pre-caching in consumers

## Architecture Notes

- GPU: 8 blocks × 64 threads = 512 threads
- CPU: 32 threads with core affinity
- Buffer: 512 elements, each padded to 4KB (page-aligned)
- Iterations: 10,000 per test
