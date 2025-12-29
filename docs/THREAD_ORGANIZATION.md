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
