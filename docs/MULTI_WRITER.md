# Multi-Writer Pattern Support

## Overview
The pattern dispatch system supports multi-writer mode, allowing exactly 4 concurrent writers (one per thread scope) to write to scope-specific buffers simultaneously.

**Note:** This document is current and describes the active multi-writer system integrated into the pattern-based dispatch framework.

## Key Features

### 1. Multi-Writer Flag
- Add `multi_writer: true` to pattern YAML to enable multi-writer mode
- System automatically allocates 4 scope-specific buffers (bufferElement_t/b/d/s)
- Validation ensures exactly 4 writers with one per scope (thread/block/device/system)

### 2. Automatic Buffer Allocation
When `multi_writer: true`, the system allocates:
- `bufferElement_t` - Thread-scope buffer
- `bufferElement_b` - Block-scope buffer  
- `bufferElement_d` - Device-scope buffer
- `bufferElement_s` - System-scope buffer

### 3. Scope-Based Routing
- **Writers**: Automatically routed to their scope-specific buffer based on `scope` field
- **Readers**: Automatically routed to buffer matching their `watch_flag` scope
- No manual buffer management required

### 4. Memory Ordering Support
All multi-writer functions support both release and relaxed orderings:
- `gpu_buffer_multi_writer_thread_release/relaxed`
- `gpu_buffer_multi_writer_block_release/relaxed`
- `gpu_buffer_multi_writer_device_release/relaxed`
- `gpu_buffer_multi_writer_system_release/relaxed`

## Validation Rules

Multi-writer patterns must satisfy:
1. Exactly 4 writers total (no more, no less)
2. Exactly 1 writer per scope (thread, block, device, system)
3. All readers must watch a scope that has a corresponding writer

## Example Configuration

```yaml
patterns:
  - name: multi_writer_test
    multi_writer: true  # Enable multi-writer mode
    
    gpu:
      blocks:
        block_0:
          # Exactly 4 writers, one per scope
          thread_0: {role: writer, scope: thread, ordering: release}
          thread_1: {role: writer, scope: block, ordering: release}
          thread_2: {role: writer, scope: device, ordering: release}
          thread_3: {role: writer, scope: system, ordering: release}
          
          # Readers automatically routed to matching scope buffers
          threads_4_7: {role: reader, ordering: acquire, watch_flag: thread}
          threads_8_11: {role: reader, ordering: acquire, watch_flag: block}
          threads_12_15: {role: reader, ordering: acquire, watch_flag: device}
          threads_16_19: {role: reader, ordering: acquire, watch_flag: system}
```

## Implementation Details

### Modified Files
1. **include/pattern_config.hpp**
   - Added `multi_writer` boolean flag to PatternConfig
   - Updated validation to enforce 4-writer constraint

2. **src/pattern_config.cpp**
   - Added YAML parsing for `multi_writer` field
   - Enhanced validation to check writer count per scope
   - Enforces exactly one writer per scope in multi-writer mode

3. **include/pattern_dispatch.cuh**
   - Added 8 multi-writer GPU functions (4 scopes × 2 orderings)
   - Added `dispatch_multi_writer()` for writer routing
   - Added `dispatch_multi_reader()` for scope-aware reader routing
   - Added `dispatch_gpu_thread_multi()` for multi-writer dispatch
   - Added `pattern_orchestrator_multi()` kernel for multi-writer patterns

4. **src/cache_invalidation_testing.cu**
   - Added conditional buffer allocation for multi-writer mode
   - Dispatches to `pattern_orchestrator_multi` when `multi_writer: true`
   - Automatic cleanup of scope-specific buffers

### Test Results
Pattern: `test_multi_writer`
- ✅ 4 concurrent writers launched successfully
- ✅ Readers receive correct scope-specific values:
  - Thread readers: 512 (1 × BUFFER_SIZE)
  - Block readers: 1024 (2 × BUFFER_SIZE)
  - Device readers: 1536 (3 × BUFFER_SIZE)
  - System readers: 2048 (4 × BUFFER_SIZE)
- ✅ Both acquire and relaxed readers work correctly
- ✅ Validation correctly rejects invalid configurations

## Usage

```bash
# Build all executables
make all

# Run multi-writer test
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -P multi_writer_test \
    -F configs/test_multi_writer.yaml \
    -m um

# List available patterns
./output/cache_invalidation_testing_DATA_SIZE_32.out \
    -F configs/test_multi_writer.yaml
```

## Technical Notes

### Wait Count Adjustment
Multi-writer functions wait for `GPU_NUM_BLOCKS * GPU_NUM_THREADS - 4` readers (instead of `-1` in single-writer mode) since there are 4 concurrent writers.

### Value Encoding
Multi-writer functions store scope-specific values:
- Thread writer: 10 → 1 (final)
- Block writer: 20 → 2 (final)
- Device writer: 30 → 3 (final)
- System writer: 40 → 4 (final)

This allows easy verification of which buffer each reader accessed.

### CPU Support
Multi-writer mode is currently GPU-only. CPU threads should be configured as dummy readers in multi-writer patterns.
