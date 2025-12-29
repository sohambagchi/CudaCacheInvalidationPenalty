# TODO: Pattern Dispatch Enhancements and Instrumentation

This document tracks tasks for improving the current pattern-based dispatch system.

## Table of Contents
1. [Documentation Updates](#documentation-updates)
2. [Timing Instrumentation](#timing-instrumentation)
3. [Pattern System Enhancements](#pattern-system-enhancements)
4. [Cleanup Tasks](#cleanup-tasks)

---

## Documentation Updates

### ✅ Completed
- [x] Created REDUNDANT.md with deprecated features
- [x] Created REFERENCE_NEW.md with current system documentation
- [x] Updated README.md to reflect pattern-based system
- [x] Updated MULTI_WRITER.md usage examples

### 🔄 In Progress
- [ ] Replace old REFERENCE.md with REFERENCE_NEW.md
- [ ] Archive or update THREAD_ORGANIZATION.md (describes old hard-coded system)
- [ ] Update TODO.md to reflect current codebase status (this file)

### 📋 Pending
- [ ] Create PATTERN_GUIDE.md with:
  - Pattern design best practices
  - Common pattern templates
  - Debugging tips for pattern validation errors
- [ ] Create EXAMPLES.md showcasing:
  - Basic release-acquire test
  - Isolated acquire per CTA
  - Cross-scope visibility tests
  - Multi-writer concurrent patterns
- [ ] Add inline code documentation:
  - Doxygen-style comments for all consumer functions
  - Document dispatch flow in detail
  - Add examples to pattern_config.hpp

---

## Timing Instrumentation

### Current Status
The framework has consumer functions but **no timing instrumentation** to measure propagation delays.

### Goals
1. Measure per-thread read/write latencies
2. Track flag propagation times
3. Identify which flag triggered first in multi-flag readers
4. Output parseable timing data for analysis

### Implementation Plan

#### Phase 1: Timing Infrastructure

**STATUS: ✅ COMPLETED** - Implemented 2025-12-29

**Data Structures:**
```c
// Per-thread GPU timing data
typedef struct gpu_timing_data {
    clock_t start_time;
    clock_t end_time;
    clock_t flag_trigger_time;
    uint32_t thread_id;
    uint32_t consumer_type;  // writer, reader_acq, reader_rlx
    uint32_t flag_type;      // thread, block, device, system
    char padding[PAGE_SIZE - sizeof(clock_t)*3 - sizeof(uint32_t)*3];
} gpu_timing_data;

// Per-thread CPU timing data
typedef struct cpu_timing_data {
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t flag_trigger_ns;
    uint32_t thread_id;
    uint32_t consumer_type;
    uint32_t flag_type;
    char padding[PAGE_SIZE - sizeof(uint64_t)*3 - sizeof(uint32_t)*3];
} cpu_timing_data;
```

✅ **Allocation:**
- Timing arrays allocated in main()
- Passed through orchestrator → dispatch → consumer chain
- GPU: cudaMalloc, CPU: std::vector or malloc

✅ **Calibration:**
- Added `calibrate_clock_overhead()` kernel
- Measures average clock64() overhead (10,000 samples)
- Overhead correction applied in CSV output

#### Phase 2: Instrument Consumer Functions

**GPU Readers:**
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_acquire(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing  // ADD THIS PARAMETER
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    timing[tid].start_time = clock64();
    timing[tid].thread_id = tid;
    
    // ... existing code ...
    
    // Time flag wait
    clock_t flag_start = clock64();
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && ...) {}
    timing[tid].flag_trigger_time = clock64();
    
    // ... read buffer ...
    
    timing[tid].end_time = clock64();
}
```

**STATUS: ✅ COMPLETED** - GPU reader timing fully implemented as of 2025-12-29
- gpu_buffer_reader_acquire and gpu_buffer_reader_relaxed instrumented
- Records start_time, flag_trigger_time, end_time
- Sets consumer_type and flag_type metadata
- Integrated into dispatch chain: orchestrator → dispatch_gpu_thread → dispatch_reader → reader functions

**GPU Writers:**
```cuda
template <typename B, typename W, typename R>
__device__ void gpu_buffer_writer_release(
    B *buffer, R *r_signal, W *w_signal, flag_s *fallback_signal,
    gpu_timing_data *timing  // ADD THIS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    timing[tid].start_time = clock64();
    
    // ... existing write code ...
    
    w_signal->flag.store(1, cuda::memory_order_release);
    timing[tid].flag_trigger_time = clock64();
    
    // ... sleep ...
    
    timing[tid].end_time = clock64();
}
```

**STATUS: 📋 FUTURE WORK** - Writer timing deferred to Phase 5
- Writers have complex multi-phase execution (thread/block/device/system scopes)
- Need to decide whether to track all 4 flag stores or just final one
- Lower priority than reader timing (readers are the critical measurement point)

**CPU Functions:** Use `std::chrono::high_resolution_clock::now()`

**STATUS: 📋 FUTURE WORK** - CPU timing deferred (Phase 6)
- CPU threads not currently launched in pattern-based system
- Will implement when CPU thread support is added

#### Phase 3: Output and Analysis

**STATUS: ✅ COMPLETED** - CSV output implemented 2025-12-29

**Output Format (CSV):**
```csv
device,block,thread,role,ordering,scope,watch_flag,start_cycles,flag_cycles,end_cycles,duration_cycles,wait_cycles,result_corrected
gpu,0,0,writer,release,device,N/A,1234567890,1234567900,1234568000,110,110,107
gpu,0,1,reader,acquire,device,device,1234567890,1234567905,1234568010,120,15,117
gpu,0,2,reader,relaxed,device,device,1234567890,1234567920,1234568025,135,30,132
```

✅ **Implementation:**
- `write_timing_csv()` function in main.cu
- Filename format: `timing_<pattern_name>_<timestamp>.csv`
- Timestamp format: YYYYMMDD_HHMMSS
- Overhead correction: duration_corrected = duration - (3 × clock_overhead)
- Skips inactive/dummy threads

**Analysis Tools:**
- 📋 **TODO:** Add to `scripts/analyze_timing.py`
- Calculate per-thread latencies
- Compare acquire vs. relaxed propagation times
- Visualize flag propagation across thread hierarchy

#### Phase 4: Regression Testing

**STATUS: 📋 FUTURE WORK**

- Add timing baseline tests
- Detect performance regressions
- Compare across memory allocators
- Track timing across different GPU architectures

---

## Phase 5: Writer Timing (Future Work)

**Deferred from Phase 2** - Writers not timed in initial implementation

### Challenges:
1. **Multi-phase execution:** Writers execute 4 sequential phases (thread/block/device/system)
2. **Timing granularity:** Should we track all 4 flag stores or just the final one?
3. **Buffer variants:** Different buffer types per scope (bufferElement_t/b/d/s)
4. **Lower priority:** Readers are the critical measurement point for propagation delays

### When to implement:
- After reader timing is validated and analyzed
- If writer-side latency becomes research focus
- When investigating write throughput vs. propagation tradeoffs

---

## Phase 6: CPU Thread Support (Future Work)

**Deferred from Phase 2** - CPU threads not currently used in pattern-based system

### Required changes:
1. Add CPU thread launching in main()
2. Instrument cpu_buffer_reader_acquire/relaxed with chrono
3. Create cpu_timing_data allocation and collection
4. Extend CSV output to include CPU timing
5. Add CPU-GPU cross-device timing analysis

### When to implement:
- When heterogeneous CPU-GPU patterns are needed
- For system-scope coherence testing
- When comparing CPU vs GPU reader latencies

---

## Pattern System Enhancements

### Configuration Features

#### ✅ Implemented
- [x] YAML-based pattern configuration
- [x] Per-thread role assignment
- [x] Multi-writer mode with validation
- [x] Block and thread range syntax
- [x] Pattern validation on load

#### 📋 Planned

**1. Pattern Composition**
```yaml
# Import and extend existing patterns
patterns:
  - name: "extended_test"
    base: "isolated_acquire_per_cta"
    
    overrides:
      gpu:
        blocks:
          block_3:
            threads_0_31: {role: reader, ordering: acquire, watch_flag: system}
```

**2. Variable Substitution**
```yaml
# Define reusable values
variables:
  main_scope: device
  reader_ordering: acquire

patterns:
  - name: "parameterized_test"
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: ${main_scope}, ordering: release}
          threads_1_31: {role: reader, ordering: ${reader_ordering}, watch_flag: ${main_scope}}
```

**3. Pattern Validation Levels**
```yaml
patterns:
  - name: "experimental_pattern"
    validation_level: permissive  # strict, permissive, none
    
    # Allow experimental configurations
```

**4. Dynamic Thread Counts**
```yaml
patterns:
  - name: "scalable_test"
    
    gpu:
      num_blocks: ${GPU_NUM_BLOCKS}  # From environment or command-line
      threads_per_block: ${GPU_THREADS_PER_BLOCK}
```

**5. Conditional Configuration**
```yaml
patterns:
  - name: "conditional_test"
    
    gpu:
      blocks:
        block_0:
          if: ${ENABLE_WRITER}
            thread_0: {role: writer, scope: device, ordering: release}
          else:
            thread_0: {role: dummy_writer}
```

### Runtime Features

**1. Pattern Profiling**
- Automatic performance profiling per pattern
- Track execution time, synchronization delays
- Output: `pattern_name_profile.json`

**2. Pattern Debugging**
- Verbose mode: `-v` flag
- Print thread dispatch decisions
- Track flag state changes
- Output synchronization timeline

**3. Pattern Verification**
- Dry-run mode: `--dry-run`
- Validate pattern without execution
- Print thread assignment table
- Check for potential deadlocks

**4. Interactive Pattern Testing**
```bash
# Interactive REPL for pattern development
./cache_invalidation_testing --interactive -F configs/test.yaml
> load pattern isolated_acquire
> show blocks
> modify block_1 thread_0 ordering=relaxed
> run
> show results
```

---

## Cleanup Tasks

### Code Organization

#### ✅ Completed
- [x] Moved legacy code to stale_code.cuh
- [x] Separated pattern dispatch into dedicated headers
- [x] Created modular pattern configuration system

#### 📋 Pending

**1. File Restructuring**
```
Current:
  include/pattern_dispatch.cuh        (GPU functions + dispatch)
  include/pattern_dispatch_cpu.hpp    (CPU functions)

Proposed:
  include/
    dispatch/
      gpu_dispatch.cuh          # GPU dispatch logic
      cpu_dispatch.hpp          # CPU dispatch logic
    consumers/
      gpu_consumers.cuh         # GPU reader/writer functions
      cpu_consumers.hpp         # CPU reader/writer functions
    config/
      pattern_config.hpp        # Configuration structures
      pattern_registry.hpp      # Registry and YAML parsing
```

**2. Remove Unused Defines**
- Remove commented CUDA_THREAD_SCOPE defines from types.hpp
- Clean up old P_H_FLAG_STORE_ORDER / C_H_FLAG_LOAD_ORDER (now in YAML)

**3. Standardize Naming**
- Consistent naming: `gpu_*` for device functions, `cpu_*` for host
- Remove misleading names (e.g., `*_diverge` functions that don't test divergence)

**4. Error Handling**
- Add CUDA error checking after all device operations
- Better error messages for pattern validation failures
- Graceful handling of invalid YAML syntax

### Testing Infrastructure

**1. Unit Tests**
- Test pattern parsing with various YAML formats
- Test pattern validation logic
- Test thread assignment calculation

**2. Integration Tests**
- Automated pattern test suite
- Run all configs/*.yaml patterns
- Verify expected result values

**3. CI/CD**
- GitHub Actions workflow
- Build all data size variants
- Run basic pattern tests
- Check documentation consistency

### Performance Optimization

**1. Constant Memory Usage**
- Currently: Full pattern copied to device constant memory
- Optimize: Only copy active block configurations

**2. Launch Configuration**
- Experiment with different block/thread counts
- Benchmark optimal configuration per pattern type

**3. Synchronization Efficiency**
- Evaluate spin-wait vs. sleep in flag polling
- Optimize fallback timeout values

---

## Long-Term Goals

### Multi-Device Support
- Extend patterns to multiple GPUs
- Test cross-device cache coherence
- Support heterogeneous GPU systems (e.g., H100 + A100)

### Advanced Patterns
- Producer-consumer queues
- Read-modify-write patterns
- Multi-stage pipeline patterns
- Hierarchical synchronization (block→device→system)

### Analysis Framework
- Statistical analysis of timing data
- Automated bottleneck detection
- Visualization dashboards
- Export to performance analysis tools (NVIDIA Nsight, VTune)

### Documentation
- Video tutorials for pattern creation
- Interactive web-based pattern builder
- Gallery of validated patterns with use cases
- Research paper references and citations

---

## Migration Notes

### Removed from Old TODO.md

The following items from the old TODO.md are **no longer applicable** because the code they referenced has been moved to `stale_code.cuh`:

- ❌ Remove simple reader/writer functions
- ❌ Remove propagation hierarchy functions  
- ❌ Remove CPU reader/writer variants
- ❌ Remove commented code blocks
- ❌ Fix bufferElement_d scope mismatch (not used in current system)
- ❌ Instrument old propagation hierarchy functions

These are documented in [REDUNDANT.md](REDUNDANT.md) for historical reference.

### Still Relevant from Old TODO

The timing implementation plan is still relevant and has been updated above to work with the current pattern dispatch system.

---

## Contributing

When working on TODOs:

1. **Mark Progress:** Update checkboxes as you complete tasks
2. **Document Changes:** Add notes on implementation decisions
3. **Update Tests:** Add tests for new features
4. **Review Docs:** Update relevant documentation files
5. **Clean Code:** Follow existing style and naming conventions

## Priority Legend

- 🔴 **Critical:** Blocks other work or affects correctness
- 🟡 **High:** Important for usability or performance
- 🟢 **Medium:** Nice to have, improves quality
- 🔵 **Low:** Future enhancement, not urgent

Current priorities:
- 🔴 Timing instrumentation (Phase 1-2)
- 🔴 Replace old REFERENCE.md
- 🟡 Pattern debugging features
- 🟡 Error handling improvements
- 🟢 Pattern composition
- 🔵 Multi-device support
