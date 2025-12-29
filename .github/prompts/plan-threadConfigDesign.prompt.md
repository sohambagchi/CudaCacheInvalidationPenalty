# CUDA Cache Invalidation Testing Framework
## Comprehensive Thread Configuration Design Guide

---

## Executive Summary

This guide provides a complete reference for designing new thread allocation patterns in the CUDA cache invalidation testing framework. The pattern system allows fine-grained control over thread roles, memory scopes, ordering semantics, and synchronization patterns to isolate specific cache coherence behaviors.

**Key Capabilities:**
- Support for 1-4 concurrent writers with scope-specific buffers
- Per-thread configuration of roles, scopes, and memory orderings
- Flexible reader distribution patterns to test propagation effects
- CPU and GPU thread orchestration with heterogeneous support
- Validation ensures patterns are logically sound before execution

---

## Design Decision Framework

### **1. Start With Your Research Question**

Ask yourself:
- What specific cache behavior am I testing?
- What hypothesis am I validating?
- What should I measure and compare?

**Examples:**
- "Does one acquire in a CTA invalidate caches for all 64 threads?"
- "Is propagation time proportional to block distance from writer?"
- "Do thread-scope and system-scope have measurably different latencies?"

---

### **2. Choose Writer Configuration**

#### **Single-Writer vs Multi-Writer Decision:**

| Use Single-Writer When... | Use Multi-Writer When... |
|---------------------------|-------------------------|
| Testing one specific scope level | Comparing all 4 scopes simultaneously |
| Measuring absolute propagation time | Need scope-specific buffers |
| Simplest case | Testing scope hierarchy |

#### **Writer Scope Selection:**

```
thread  → Same warp only (32 threads)
block   → Same CTA only (64 threads)  
device  → All GPU blocks (512 threads) ← MOST COMMON
system  → CPU + GPU (heterogeneous)
```

#### **Writer Placement Strategy:**
- **Block 0, Thread 0** - Standard, predictable scheduling
- **Different block** - Test inter-block propagation
- **CPU thread** - Test CPU→GPU propagation (requires `scope: system`)

---

### **3. Design Reader Distribution**

#### **Core Patterns:**

**Pattern A: Uniform (Baseline)**
```yaml
blocks_1_7:
  all_threads: {role: reader, ordering: acquire, watch_flag: device}
```
*Use for:* Establishing baseline synchronized timing

**Pattern B: Mixed Ordering (Primary Experiment)**
```yaml
block_1:
  thread_0: {role: reader, ordering: acquire, watch_flag: device}
  threads_1_63: {role: reader, ordering: relaxed, watch_flag: device}
```
*Use for:* **Testing cache invalidation spillover** - Key experiment!

**Pattern C: Control Group**
```yaml
block_2:
  all_threads: {role: reader, ordering: relaxed, watch_flag: device}
```
*Use for:* Comparison baseline without any acquire

**Pattern D: Stratified by Scope**
```yaml
block_1: {threads_0_15: {role: reader, watch_flag: thread}}
block_2: {threads_0_15: {role: reader, watch_flag: block}}
block_3: {threads_0_15: {role: reader, watch_flag: device}}
block_4: {threads_0_15: {role: reader, watch_flag: system}}
```
*Use for:* Multi-writer mode to compare scope propagation speeds

---

### **4. Dummy Thread Strategy**

**Purpose:**
- Maintain consistent GPU occupancy (avoid scheduler artifacts)
- Generate realistic memory traffic
- Create cache pressure

**Guidelines:**
- **25% dummy** - Light background load
- **50% dummy** - Moderate (realistic multi-app)
- **75% dummy** - Heavy contention stress test

**Fill remaining threads:**
```yaml
threads_20_63: {role: dummy_reader}
blocks_5_7: {all_threads: {role: dummy_reader}}
```

---

### **5. Validation Rules (Must Satisfy)**

✅ **Multi-writer mode:**
- Exactly 4 writers (one per scope: thread/block/device/system)
- Set `multi_writer: true`

✅ **Flag coverage:**
- Every reader's `watch_flag` must have matching writer `scope`

✅ **Thread counts:**
- GPU: 8 blocks × 64 threads = 512 total
- CPU: 32 threads
- No overlapping ranges

✅ **System scope:**
- If using `scope: system` or `watch_flag: system`, CPU must participate

---

## Pattern Templates Library

### **Template 1: Minimal Test**
```yaml
patterns:
  - name: minimal_sync
    description: Simplest writer-reader test
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          thread_1: {role: reader, ordering: acquire, watch_flag: device}
          threads_2_63: {role: dummy_reader}
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      threads:
        all_threads: {role: inactive}
```

### **Template 2: Cache Invalidation Spillover** ⭐ **Primary Experiment**
```yaml
patterns:
  - name: isolated_acquire
    description: Test if one acquire affects other relaxed loads in CTA
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        # TEST: 1 acquire + 63 relaxed
        block_1:
          thread_0: {role: reader, ordering: acquire, watch_flag: device}
          threads_1_63: {role: reader, ordering: relaxed, watch_flag: device}
        
        # CONTROL: All relaxed (no acquire)
        block_2:
          all_threads: {role: reader, ordering: relaxed, watch_flag: device}
        
        blocks_3_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      threads:
        all_threads: {role: inactive}
```

**Hypothesis:** Do block 1's 63 relaxed threads see cache invalidation from thread 0's acquire?

**Compare:** Block 1 relaxed timing vs Block 2 relaxed timing

### **Template 3: Multi-Writer Scope Comparison**
```yaml
patterns:
  - name: scope_hierarchy
    description: Compare all 4 scope levels simultaneously
    multi_writer: true
    
    gpu:
      blocks:
        block_0:
          # 4 writers (one per scope)
          thread_0: {role: writer, scope: thread, ordering: release}
          thread_1: {role: writer, scope: block, ordering: release}
          thread_2: {role: writer, scope: device, ordering: release}
          thread_3: {role: writer, scope: system, ordering: release}
          
          # Scope-specific readers
          threads_4_7: {role: reader, ordering: acquire, watch_flag: thread}
          threads_8_11: {role: reader, ordering: acquire, watch_flag: block}
          threads_12_15: {role: reader, ordering: acquire, watch_flag: device}
          threads_16_19: {role: reader, ordering: acquire, watch_flag: system}
          threads_20_63: {role: dummy_reader}
        
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      threads:
        all_threads: {role: dummy_reader}
```

**Expected Results:**
- Thread readers: 512 (1 × 512)
- Block readers: 1024 (2 × 512)
- Device readers: 1536 (3 × 512)
- System readers: 2048 (4 × 512)

### **Template 4: Warp-Level Test**
```yaml
patterns:
  - name: warp_mixing
    description: Test acquire effect at warp boundaries (32 threads)
    
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        block_1:
          # Warp 0: 1 acquire + 31 relaxed
          thread_0: {role: reader, ordering: acquire, watch_flag: device}
          threads_1_31: {role: reader, ordering: relaxed, watch_flag: device}
          
          # Warp 1: All relaxed (control)
          threads_32_63: {role: reader, ordering: relaxed, watch_flag: device}
```

**Question:** Does acquire in warp 0 affect warp 1?

### **Template 5: CPU-GPU Heterogeneous**
```yaml
patterns:
  - name: cpu_to_gpu
    description: Test system-scope CPU→GPU propagation
    
    gpu:
      blocks:
        block_0:
          threads_0_31: {role: reader, ordering: acquire, watch_flag: system}
          threads_32_63: {role: dummy_reader}
        blocks_1_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      threads:
        thread_0: {role: writer, scope: system, ordering: release}
        threads_1_15: {role: reader, ordering: acquire, watch_flag: system}
        threads_16_31: {role: dummy_reader}
```

---

## YAML Configuration Syntax Reference

### Thread Range Syntax

**Single Thread:**
```yaml
thread_0: {role: writer, scope: device, ordering: release}
```

**Thread Range:**
```yaml
threads_4_7: {role: reader, ordering: acquire, watch_flag: thread}
```

**All Threads:**
```yaml
all_threads: {role: dummy_reader}
```

### Block Range Syntax

**Single Block:**
```yaml
block_0:
  thread_0: {role: writer, ...}
  threads_1_63: {role: reader, ...}
```

**Block Range:**
```yaml
blocks_2_7:
  all_threads: {role: dummy_reader}
```

### Thread Configuration Fields

**Required for all roles:**
- `role`: writer | reader | dummy_reader | dummy_writer | inactive

**Required for writers:**
- `scope`: thread | block | device | system
- `ordering`: release | relaxed

**Required for readers:**
- `ordering`: acquire | relaxed
- `watch_flag`: thread | block | device | system (which flag scope to wait on)

**Dummy threads and inactive:**
- No additional fields required

---

## Thread Roles and Behaviors

### ThreadRole Types

**WRITER**
- Waits for all readers to signal ready
- Writes buffer values (10, 20, 30, 40 for single-writer; 1, 2, 3, 4 for multi-writer)
- Sets scope-specific flags progressively
- Single-writer mode: One writer, readers wait on different scope flags
- Multi-writer mode: 4 writers, each with scope-specific buffer

**READER**
- Signals ready via r_signal.fetch_add(1)
- Waits on specific watch_flag with acquire or relaxed ordering
- Reads entire buffer (BUFFER_SIZE elements)
- Stores sum of buffer values in results array

**DUMMY_READER**
- Generates background memory traffic
- No synchronization with writers
- Not included in timing measurements
- Use to simulate realistic concurrent workload

**DUMMY_WRITER**
- Generates background write traffic
- No flag setting for readers
- Not included in timing measurements

**INACTIVE**
- Does nothing
- Use for CPU threads in GPU-only tests

---

## Memory Scopes and Orderings

### ThreadScope Levels

| Scope | Coherence Domain | Use When |
|-------|------------------|----------|
| `thread` | Single warp (32 threads) | Testing warp-level effects |
| `block` | Single CTA (64 threads) | Testing block-level propagation |
| `device` | All GPU blocks (512 threads) | Testing cross-CTA effects (MOST COMMON) |
| `system` | CPU + GPU | Testing heterogeneous coherence |

### MemoryOrdering Options

**For Writers:**
- `release` - Standard synchronization (use for most tests)
- `relaxed` - No synchronization guarantees (experimental)

**For Readers:**
- `acquire` - Synchronizes with release writer
- `relaxed` - No synchronization (tests hardware-only coherence)

**Key Experiment:** Do relaxed readers benefit from nearby acquire reader's cache invalidation?

---

## Design Decision Matrix

| Test Goal | Writer Mode | Writer Scope | Reader Pattern | Dummy % |
|-----------|------------|--------------|----------------|---------|
| Basic sync | Single | device | Uniform acquire | 75% |
| Cache spillover | Single | device | 1 acq + 63 rlx | 75% |
| Scope comparison | Multi | all 4 | Stratified by scope | 75% |
| Warp effects | Single | device | Split at warp boundary | 50% |
| CPU-GPU coherence | Single | system | CPU + GPU | 75% |
| Distance effects | Single | device | Readers at varying block distances | 50% |

---

## Analysis Workflow

### **Step 1: Design with Controls**
Always include:
1. **Test group** (what you're measuring)
2. **Positive control** (should show effect)
3. **Negative control** (should NOT show effect)

### **Step 2: Minimize Variables**
Change only ONE thing between patterns:
- Same pattern, different ordering
- Same pattern, different distribution
- Same pattern, different scope

### **Step 3: Expected Results**

**Single-Writer Values:**
- Thread flag readers: 10 × 512 = 5,120
- Block flag readers: 20 × 512 = 10,240
- Device flag readers: 30 × 512 = 15,360
- System flag readers: 40 × 512 = 20,480

**Multi-Writer Values:**
- Thread buffer readers: 1 × 512 = 512
- Block buffer readers: 2 × 512 = 1,024
- Device buffer readers: 3 × 512 = 1,536
- System buffer readers: 4 × 512 = 2,048

---

## Common Mistakes to Avoid

❌ Multi-writer without exactly 4 writers  
❌ Reader `watch_flag` doesn't match any writer `scope`  
❌ Changing multiple variables between patterns  
❌ No control groups  
❌ Missing dummy threads (unrealistic 100% active)  
❌ System scope without CPU participation  
❌ Overlapping thread ranges within a block  
❌ Vague pattern names like "test1", "test2"  

---

## Pre-Flight Checklist

Before running:
- [ ] Clear test goal written down
- [ ] Writer placement decided (single or multi)
- [ ] Reader distribution follows a pattern template
- [ ] Control groups included
- [ ] Dummy threads fill remaining slots (~75%)
- [ ] Validation rules satisfied
- [ ] Expected results calculated
- [ ] Baseline pattern exists for comparison

---

## Complete Example Workflow

**Goal:** Test if a single acquire load in a CTA causes cache invalidation for other relaxed loads in the same CTA.

### Step 1: Fill Design Worksheet

```
Pattern Name: cta_invalidation_test
Test Goal: Measure if one acquire in CTA affects relaxed threads

Writer Configuration:
  [X] Single-writer  [ ] Multi-writer
  Writer scope: device
  Writer ordering: [X] release  [ ] relaxed
  Writer location: Block 0, Thread 0

Reader Configuration:
  Total readers: 128 (2 blocks × 64 threads)
  Acquire readers: 2 (one per test block)
  Relaxed readers: 126
  Watch flag: device
  Distribution: Cross-block (2 blocks)

Dummy Configuration:
  Dummy readers: 384 (6 blocks × 64)
  Purpose: Background memory traffic

Expected Result:
  All readers see 15360 (30 × 512) eventually
  Question: Do relaxed readers in block 1 have similar timing to block 2?

Control Groups:
  Block 1: 1 acquire + 63 relaxed (test)
  Block 2: 0 acquire + 64 relaxed (control)
```

### Step 2: Write YAML

```yaml
patterns:
  - name: cta_invalidation_test
    description: Test if one acquire in CTA causes cache invalidation for relaxed loads
    multi_writer: false
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        # Block 0: Writer only
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        # Block 1: TEST - 1 acquire + 63 relaxed
        block_1:
          thread_0: {role: reader, ordering: acquire, watch_flag: device}
          threads_1_63: {role: reader, ordering: relaxed, watch_flag: device}
        
        # Block 2: CONTROL - All relaxed
        block_2:
          all_threads: {role: reader, ordering: relaxed, watch_flag: device}
        
        # Blocks 3-7: Background traffic
        blocks_3_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      threads:
        all_threads: {role: inactive}
```

### Step 3: Validate

Check:
- ✅ Single writer present
- ✅ Writer scope is device
- ✅ All readers watching device flag
- ✅ No overlapping thread ranges
- ✅ Control group (block 2) present
- ✅ Dummy threads for background traffic

### Step 4: Run

```bash
make pattern
./output/cache_invalidation_testing_pattern_CUDA_THREAD_SCOPE_DEVICE_DATA_SIZE_32.out \
    -P cta_invalidation_test \
    -F configs/cta_test.yaml \
    -m um
```

### Step 5: Analyze Results

**Hypothesis A: Hardware invalidates entire CTA**
- Block 1 relaxed threads should have similar timing to acquire thread
- Block 2 relaxed threads should be slower (no acquire to trigger invalidation)

**Hypothesis B: Hardware invalidates only specific cache lines**
- Block 1 relaxed threads should have similar timing to Block 2
- Only acquire thread has fast access

**Data to Compare:**
- Block 1 Thread 0 (acquire) timing
- Block 1 Threads 1-63 (relaxed) timing average
- Block 2 Threads 0-63 (relaxed) timing average

---

## Advanced Techniques

### Parametric Patterns

Create pattern families by varying one parameter:

Example: Acquire density series
- Pattern A: 1 acquire per 64 threads
- Pattern B: 4 acquire per 64 threads
- Pattern C: 16 acquire per 64 threads
- Pattern D: 64 acquire per 64 threads

Plot performance vs acquire density to find inflection points.

### Multi-Block Replication

Replicate same configuration across blocks for statistical power:

```yaml
# Replicate test configuration across blocks 1-7
blocks_1_7:
  thread_0: {role: reader, ordering: acquire, watch_flag: device}
  threads_1_63: {role: reader, ordering: relaxed, watch_flag: device}
```

Advantage: 7 independent measurements of same configuration.

### Thread Position Testing

Test if thread position within warp/block matters:

```yaml
# Variant A: Acquire at thread 0 (start of warp)
# Variant B: Acquire at thread 16 (middle of warp)
# Variant C: Acquire at thread 31 (end of warp)
```

Question: Does acquire position within warp affect invalidation pattern?

---

## Summary and Best Practices

### Golden Rules

1. **Start Simple:** Begin with minimal patterns, add complexity incrementally
2. **Use Control Groups:** Always have baseline and comparison patterns
3. **Validate Early:** Check validation before complex configurations
4. **Match Scopes:** Default to matching reader watch_flag with writer scope
5. **Document Intent:** Use clear descriptions explaining what you're testing
6. **Replicate:** Use multiple blocks to get statistical measurements
7. **Dummy Threads:** Include 50-75% dummy threads for realistic tests
8. **Name Clearly:** Use descriptive pattern names indicating test goal

### Recommended Development Workflow

1. **Define Goal:** Write 1-2 sentence description of what you're testing
2. **Fill Worksheet:** Complete the pattern design worksheet
3. **Pick Template:** Start from closest template
4. **Modify:** Adjust thread counts and configurations
5. **Validate:** Check against validation rules
6. **Test:** Run with minimal config first
7. **Expand:** Add complexity (more blocks, dummy threads)
8. **Compare:** Create companion patterns with one variable changed
9. **Document:** Add comments explaining non-obvious choices
10. **Iterate:** Refine based on results

---

**Key Insight:** The most valuable patterns test the "cache invalidation spillover" hypothesis - does one acquire thread's cache invalidation affect other relaxed threads in the same CTA? Use Template 2 as your starting point for this primary experiment.
