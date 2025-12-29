# Implementation Plan: Template Dispatch Pattern System

**Option B: Per-Thread Ordering Control via Explicit Function Variants**

This document provides a comprehensive, actionable plan for implementing a thread pattern specification system that enables per-thread ordering control (acquire/relaxed) within a single binary.

---

## Phase 1: Data Structures & YAML Schema (Week 1, Days 1-2)

### YAML Schema Design

```yaml
# thread_patterns.yaml
patterns:
  - name: "isolated_acquire_per_cta"
    description: "Test if single acquire in CTA affects other relaxed loads"
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      # Simple block specification
      blocks:
        block_0:
          thread_0:
            role: writer
            scope: device
            ordering: release
          threads_1_63:
            role: dummy_reader
        
        block_1:
          thread_0:
            role: reader
            scope: device
            ordering: acquire
            watch_flag: device
          threads_1_63:
            role: reader
            scope: thread
            ordering: relaxed
            watch_flag: thread
        
        blocks_2_7:
          all_threads:
            role: dummy_reader
    
    cpu:
      num_threads: 32
      all_threads:
        role: dummy_reader
```

### C++ Data Structures

Create new file `pattern_config.h`:

```cpp
#ifndef PATTERN_CONFIG_H
#define PATTERN_CONFIG_H

#include <string>
#include <map>

// Enums for configuration
enum class ThreadRole : uint8_t {
    INACTIVE = 0,
    WRITER = 1,
    READER = 2,
    DUMMY_READER = 3,
    DUMMY_WRITER = 4
};

enum class ThreadScope : uint8_t {
    THREAD = 0,
    BLOCK = 1,
    DEVICE = 2,
    SYSTEM = 3
};

enum class MemoryOrdering : uint8_t {
    RELAXED = 0,
    ACQUIRE = 1,
    RELEASE = 2,
    ACQ_REL = 3
};

// Per-thread configuration (compact: 4 bytes)
struct ThreadConfig {
    ThreadRole role;
    ThreadScope scope;
    MemoryOrdering ordering;
    ThreadScope watch_flag;  // Which flag scope to observe
    
    ThreadConfig() 
        : role(ThreadRole::INACTIVE)
        , scope(ThreadScope::DEVICE)
        , ordering(MemoryOrdering::RELAXED)
        , watch_flag(ThreadScope::DEVICE) {}
};

// Pattern storage
struct PatternConfig {
    std::string name;
    std::string description;
    
    // GPU: [block_id][thread_id]
    ThreadConfig gpu_threads[8][64];
    
    // CPU: [core_id]
    ThreadConfig cpu_threads[32];
    
    int gpu_num_blocks;
    int gpu_threads_per_block;
    int cpu_num_threads;
};

// Pattern registry
class PatternRegistry {
private:
    std::map<std::string, PatternConfig> patterns_;
    
public:
    bool load_from_yaml(const std::string& yaml_path);
    const PatternConfig* get_pattern(const std::string& name) const;
    void list_patterns() const;
};

// Global instance
extern PatternRegistry g_pattern_registry;

#endif // PATTERN_CONFIG_H
```

### Implementation Task List

**File: `pattern_config.cpp`**
- [ ] Implement YAML parsing (use yaml-cpp or simple custom parser)
- [ ] Parse block ranges (`block_0`, `blocks_2_7`)
- [ ] Parse thread ranges (`thread_0`, `threads_1_63`, `all_threads`)
- [ ] Validate configurations (thread counts, role consistency)
- [ ] Implement `PatternRegistry::load_from_yaml()`
- [ ] Add error reporting for invalid patterns

**Dependencies:**
```bash
# Add to Makefile
LIBS += -lyaml-cpp
# Or implement simple parser without dependencies
```

---

## Phase 2: Refactor Functions with Ordering Variants (Week 1, Days 3-5)

### Current Function Signatures to Refactor

**GPU Functions (in `gpu_kernels.cuh`):**

Current:
```cpp
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_propagation_hierarchy(
    B *buffer, bufferElement_na *results, 
    R *r_signal, W *w_signal, flag_s *fallback_signal
)
```

**Create Explicit Ordering Variants:**

Add to `gpu_kernels.cuh`:

```cpp
// ============= READER VARIANTS =============

// Acquire variant
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_acquire(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    #ifdef CONSUMERS_CACHE
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    #endif
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // ACQUIRE LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_acquire) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    printf("B[%d] T[%d] ACQ Result %d\n", blockIdx.x, threadIdx.x, result);
}

// Relaxed variant
template <typename B, typename W, typename R>
__device__ void gpu_buffer_reader_relaxed(
    B *buffer, bufferElement_na *results,
    R *r_signal, W *w_signal, flag_s *fallback_signal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint result = 0;
    
    #ifdef CONSUMERS_CACHE
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    #endif
    
    results[tid].data = result;
    r_signal->flag.fetch_add(1, cuda::memory_order_relaxed);
    result = 0;
    
    // RELAXED LOAD on flag
    while(w_signal->flag.load(cuda::memory_order_relaxed) == 0 && 
          fallback_signal->flag.load(cuda::memory_order_relaxed) < 3) {
        // Wait
    }
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        result += buffer[i].data.load(cuda::memory_order_relaxed);
    }
    
    results[tid].data = result;
    printf("B[%d] T[%d] RLX Result %d\n", blockIdx.x, threadIdx.x, result);
}

// ============= WRITER VARIANTS =============

// Release variant
__device__ void gpu_buffer_writer_release(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("GPU Writer (Release)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_release);  // RELEASE
    
    cudaSleep(10000000000);
    fallback_signal->flag.store(4, cuda::memory_order_release);
}

// Relaxed variant
__device__ void gpu_buffer_writer_relaxed(
    bufferElement *buffer, flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    printf("GPU Writer (Relaxed)\n");
    
    while (r_signal->flag.load(cuda::memory_order_relaxed) != 
           GPU_NUM_BLOCKS * GPU_NUM_THREADS - 1) {}
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(10, cuda::memory_order_relaxed);
    }
    w_t_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(20, cuda::memory_order_relaxed);
    }
    w_b_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(30, cuda::memory_order_relaxed);
    }
    w_d_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i].data.store(40, cuda::memory_order_relaxed);
    }
    w_s_signal->flag.store(1, cuda::memory_order_relaxed);  // RELAXED
    
    cudaSleep(10000000000);
    fallback_signal->flag.store(4, cuda::memory_order_release);
}
```

**CPU Functions - Same Pattern:**

In `cpu_functions.h`, create:
- `cpu_buffer_reader_acquire()`
- `cpu_buffer_reader_relaxed()`
- `cpu_buffer_writer_release()`
- `cpu_buffer_writer_relaxed()`

---

## Phase 3: Dispatch Logic (Week 2, Days 1-2)

### Device-Side Constant Memory

Add to `gpu_kernels.cuh`:

```cpp
// Device-side pattern configuration
__constant__ ThreadConfig d_pattern_gpu[8][64];
```

### Dispatch Functions

Add to `gpu_kernels.cuh`:

```cpp
__device__ void dispatch_gpu_thread(
    int bid, int tid,
    bufferElement *buffer,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    // Fetch config from constant memory
    ThreadConfig cfg = d_pattern_gpu[bid][tid];
    
    switch (cfg.role) {
        case ThreadRole::WRITER:
            dispatch_writer(cfg, buffer, r_signal, 
                          w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                          fallback_signal);
            break;
            
        case ThreadRole::READER:
            dispatch_reader(cfg, buffer, results, r_signal,
                          w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                          fallback_signal);
            break;
            
        case ThreadRole::DUMMY_READER:
            gpu_dummy_reader_worker_propagation(dummy_buffer, results, r_signal);
            break;
            
        case ThreadRole::DUMMY_WRITER:
            gpu_dummy_writer_worker_propagation(dummy_buffer, r_signal);
            break;
            
        case ThreadRole::INACTIVE:
        default:
            // Do nothing
            break;
    }
}

__device__ void dispatch_writer(
    ThreadConfig cfg,
    bufferElement *buffer,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    if (cfg.ordering == MemoryOrdering::RELEASE) {
        gpu_buffer_writer_release(buffer, r_signal,
                                 w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                 fallback_signal);
    } else {  // RELAXED
        gpu_buffer_writer_relaxed(buffer, r_signal,
                                 w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                                 fallback_signal);
    }
}

__device__ void dispatch_reader(
    ThreadConfig cfg,
    bufferElement *buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    // Select flag pointer based on watch_flag scope
    void* flag_ptr = nullptr;
    switch (cfg.watch_flag) {
        case ThreadScope::THREAD: flag_ptr = w_t_signal; break;
        case ThreadScope::BLOCK: flag_ptr = w_b_signal; break;
        case ThreadScope::DEVICE: flag_ptr = w_d_signal; break;
        case ThreadScope::SYSTEM: flag_ptr = w_s_signal; break;
    }
    
    // Dispatch to acquire or relaxed variant based on ordering
    if (cfg.ordering == MemoryOrdering::ACQUIRE) {
        // Dispatch to acquire reader with appropriate flag type
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                gpu_buffer_reader_acquire(buffer, results, r_signal,
                                        (flag_t*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::BLOCK:
                gpu_buffer_reader_acquire(buffer, results, r_signal,
                                        (flag_b*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::DEVICE:
                gpu_buffer_reader_acquire(buffer, results, r_signal,
                                        (flag_d*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::SYSTEM:
                gpu_buffer_reader_acquire(buffer, results, r_signal,
                                        (flag_s*)flag_ptr, fallback_signal);
                break;
        }
    } else {  // RELAXED
        switch (cfg.watch_flag) {
            case ThreadScope::THREAD:
                gpu_buffer_reader_relaxed(buffer, results, r_signal,
                                        (flag_t*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::BLOCK:
                gpu_buffer_reader_relaxed(buffer, results, r_signal,
                                        (flag_b*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::DEVICE:
                gpu_buffer_reader_relaxed(buffer, results, r_signal,
                                        (flag_d*)flag_ptr, fallback_signal);
                break;
            case ThreadScope::SYSTEM:
                gpu_buffer_reader_relaxed(buffer, results, r_signal,
                                        (flag_s*)flag_ptr, fallback_signal);
                break;
        }
    }
}
```

### Generic Orchestrator Kernel

Add to `gpu_kernels.cuh`:

```cpp
__global__ void pattern_orchestrator(
    bufferElement *buffer,
    bufferElement *dummy_buffer,
    bufferElement_na *results,
    flag_d *r_signal,
    flag_t *w_t_signal, flag_b *w_b_signal,
    flag_d *w_d_signal, flag_s *w_s_signal,
    flag_s *fallback_signal
) {
    dispatch_gpu_thread(blockIdx.x, threadIdx.x,
                       buffer, dummy_buffer, results,
                       r_signal, w_t_signal, w_b_signal,
                       w_d_signal, w_s_signal, fallback_signal);
}
```

---

## Phase 4: Main Integration (Week 2, Days 3-5)

### Modify main.cu

```cpp
#include "pattern_config.h"

// Global pattern registry
PatternRegistry g_pattern_registry;

int main(int argc, char* argv[]) {
    // ... existing arg parsing ...
    
    std::string pattern_name = "";
    std::string pattern_file = "thread_patterns.yaml";
    
    // Add new command-line flags
    int opt;
    while ((opt = getopt(argc, argv, "m:r:w:pP:F:")) != -1) {
        switch (opt) {
            // ... existing cases ...
            case 'P':
                pattern_name = optarg;
                break;
            case 'F':
                pattern_file = optarg;
                break;
        }
    }
    
    // Load patterns
    std::cout << "[INFO] Loading patterns from: " << pattern_file << std::endl;
    if (!g_pattern_registry.load_from_yaml(pattern_file)) {
        std::cerr << "[ERROR] Failed to load patterns" << std::endl;
        return 1;
    }
    
    // If no pattern specified, list available and exit
    if (pattern_name.empty()) {
        std::cout << "[INFO] Available patterns:" << std::endl;
        g_pattern_registry.list_patterns();
        return 0;
    }
    
    // Get pattern
    const PatternConfig* pattern = g_pattern_registry.get_pattern(pattern_name);
    if (!pattern) {
        std::cerr << "[ERROR] Pattern not found: " << pattern_name << std::endl;
        g_pattern_registry.list_patterns();
        return 1;
    }
    
    std::cout << "[INFO] Using pattern: " << pattern->name << std::endl;
    std::cout << "[INFO] Description: " << pattern->description << std::endl;
    
    // Copy pattern to device constant memory
    cudaMemcpyToSymbol(d_pattern_gpu, pattern->gpu_threads,
                       sizeof(ThreadConfig) * 8 * 64);
    
    // ... existing buffer allocation ...
    
    // Launch pattern orchestrator instead of old orchestrator
    pattern_orchestrator<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(
        buffer, dummy_buffer, result_g,
        r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
        w_fb_signal
    );
    
    // ... rest of execution ...
}
```

### Update Makefile

```makefile
# Add pattern-aware building
PATTERN_AWARE = -DPATTERN_DISPATCH

NVCC_FLAGS += $(PATTERN_AWARE)

# Optional: add yaml-cpp if using that library
LIBS += -lyaml-cpp
```

---

## Implementation Checklist

### Week 1: Core Infrastructure

**Day 1-2: Data Structures**
- [ ] Create `pattern_config.h` with enums and structs
- [ ] Create `pattern_config.cpp` with YAML parser
- [ ] Test loading a simple pattern from YAML
- [ ] Implement `PatternRegistry::list_patterns()`

**Day 3-5: Function Variants**
- [ ] Create `gpu_buffer_reader_acquire()`
- [ ] Create `gpu_buffer_reader_relaxed()`
- [ ] Create `gpu_buffer_writer_release()`
- [ ] Create `gpu_buffer_writer_relaxed()`
- [ ] Test each variant individually
- [ ] Create corresponding CPU variants

### Week 2: Dispatch & Integration

**Day 1-2: Dispatch Logic**
- [ ] Add `__constant__ d_pattern_gpu[8][64]`
- [ ] Implement `dispatch_gpu_thread()`
- [ ] Implement `dispatch_reader()`
- [ ] Implement `dispatch_writer()`
- [ ] Create `pattern_orchestrator` kernel

**Day 3-5: Main Integration & Testing**
- [ ] Add `-P` and `-F` flags to main.cu
- [ ] Integrate pattern loading at startup
- [ ] Test with simple all-dummy pattern
- [ ] Test with mixed reader/writer pattern
- [ ] Test with your "isolated acquire" pattern
- [ ] Compare results with old hard-coded orchestrator

---

## Testing Strategy

### Test Pattern 1: All Dummy (Sanity Check)

```yaml
patterns:
  - name: "test_all_dummy"
    gpu:
      blocks:
        blocks_0_7:
          all_threads: {role: dummy_reader}
    cpu:
      all_threads: {role: dummy_reader}
```

**Expected:** All threads run dummy readers, no crashes.

### Test Pattern 2: Single Writer, Single Reader

```yaml
patterns:
  - name: "test_simple"
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, ordering: release}
          thread_1: {role: reader, scope: device, ordering: acquire, watch_flag: device}
          threads_2_63: {role: dummy_reader}
        blocks_1_7:
          all_threads: {role: dummy_reader}
```

**Expected:** Thread 1 observes value written by Thread 0.

### Test Pattern 3: Mixed Ordering (Your Goal)

```yaml
patterns:
  - name: "test_isolated_acquire"
    gpu:
      blocks:
        block_0:
          thread_0: {role: writer, ordering: release}
          threads_1_63: {role: dummy_reader}
        block_1:
          thread_0: {role: reader, scope: device, ordering: acquire, watch_flag: device}
          threads_1_63: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
```

**Expected:** Thread 0 uses acquire, threads 1-63 use relaxed, visible in output.

---

## Usage Examples

```bash
# List available patterns
./cache_invalidation_testing.out

# Run specific pattern
./cache_invalidation_testing.out -m malloc -r gpu -w gpu \
    -P isolated_acquire_per_cta

# Use custom pattern file
./cache_invalidation_testing.out -m malloc -r gpu -w gpu \
    -P my_custom_pattern \
    -F my_patterns.yaml

# Combine with other existing flags
./cache_invalidation_testing.out -m numa_device -r gpu -w gpu \
    -P scope_hierarchy \
    -F thread_patterns.yaml
```

---

## Example Pattern Definitions

### Pattern 1: Isolated Acquire in CTA

```yaml
patterns:
  - name: "isolated_acquire_per_cta"
    description: "Test if single acquire in CTA affects other relaxed loads"
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        # Block 0: Writer
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        # Block 1: Device-scope acquire + thread-scope relaxed
        block_1:
          thread_0: {role: reader, scope: device, ordering: acquire, watch_flag: device}
          threads_1_63: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        # Block 2: Device-scope relaxed + thread-scope relaxed (control)
        block_2:
          thread_0: {role: reader, scope: device, ordering: relaxed, watch_flag: device}
          threads_1_63: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        # Block 3: System-scope relaxed + thread-scope relaxed
        block_3:
          thread_0: {role: reader, scope: system, ordering: relaxed, watch_flag: system}
          threads_1_63: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        # Block 4: System-scope acquire + device-scope relaxed
        block_4:
          thread_0: {role: reader, scope: system, ordering: acquire, watch_flag: system}
          threads_1_63: {role: reader, scope: device, ordering: relaxed, watch_flag: device}
        
        # Blocks 5-7: Dummy load for background traffic
        blocks_5_7:
          thread_0: {role: dummy_writer}
          threads_1_63: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      all_threads: {role: dummy_reader}
```

### Pattern 2: Scope Hierarchy Comparison

```yaml
patterns:
  - name: "scope_hierarchy_systematic"
    description: "Each CTA observes different scope flag, same ordering"
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        # Each block watches a different scope level
        block_1:
          all_threads: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        block_2:
          all_threads: {role: reader, scope: block, ordering: relaxed, watch_flag: block}
        
        block_3:
          all_threads: {role: reader, scope: device, ordering: relaxed, watch_flag: device}
        
        block_4:
          all_threads: {role: reader, scope: system, ordering: relaxed, watch_flag: system}
        
        # Blocks 5-7: Dummy background
        blocks_5_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      all_threads: {role: dummy_reader}
```

### Pattern 3: Warp-Level Mixing

```yaml
patterns:
  - name: "warp_level_acquire_effect"
    description: "Test acquire effect within warp (32 threads)"
    
    gpu:
      num_blocks: 8
      threads_per_block: 64
      
      blocks:
        block_0:
          thread_0: {role: writer, scope: device, ordering: release}
          threads_1_63: {role: dummy_reader}
        
        # Block 1: Warp 0 has 1 acquire + 31 relaxed
        block_1:
          thread_0: {role: reader, scope: device, ordering: acquire, watch_flag: device}
          threads_1_31: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
          # Warp 1: All relaxed (control group)
          threads_32_63: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        # Block 2: All relaxed (cross-CTA control)
        block_2:
          all_threads: {role: reader, scope: thread, ordering: relaxed, watch_flag: thread}
        
        # Blocks 3-7: Dummy
        blocks_3_7:
          all_threads: {role: dummy_reader}
    
    cpu:
      num_threads: 32
      all_threads: {role: dummy_reader}
```

---

## Key Advantages of This Approach

✅ **Per-thread ordering control** - Thread 0 can use acquire, Thread 1 can use relaxed  
✅ **Single binary** - No need for multiple executables  
✅ **Fast iteration** - Edit YAML, rerun (no recompile)  
✅ **Type-safe dispatch** - Compiler instantiates all variants  
✅ **Readable code** - Explicit `_acquire` / `_relaxed` function names  
✅ **Backward compatible** - Old orchestrators still work  

This gives you exactly what you need: **"1 device-acquire + 63 thread-relaxed in same CTA"**

---

## Alternative: Simple Parser (No yaml-cpp Dependency)

If you want to avoid the yaml-cpp dependency, here's a simple parser implementation:

```cpp
// pattern_parser.cpp - Simple YAML-like parser
#include "pattern_config.h"
#include <fstream>
#include <sstream>
#include <algorithm>

struct ParseContext {
    std::string current_pattern_name;
    std::string current_block;
    int indent_level;
    bool in_gpu_section;
    bool in_cpu_section;
};

bool PatternRegistry::load_from_yaml(const std::string& yaml_path) {
    std::ifstream file(yaml_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << yaml_path << std::endl;
        return false;
    }
    
    PatternConfig current_pattern;
    ParseContext ctx;
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Parse based on indentation and content
        if (line.find("- name:") == 0) {
            // New pattern
            if (!current_pattern.name.empty()) {
                patterns_[current_pattern.name] = current_pattern;
            }
            current_pattern = PatternConfig();
            current_pattern.name = parse_value(line);
        }
        else if (line.find("description:") == 0) {
            current_pattern.description = parse_value(line);
        }
        else if (line == "gpu:") {
            ctx.in_gpu_section = true;
            ctx.in_cpu_section = false;
        }
        else if (line == "cpu:") {
            ctx.in_cpu_section = true;
            ctx.in_gpu_section = false;
        }
        // ... continue parsing blocks and threads ...
    }
    
    // Add last pattern
    if (!current_pattern.name.empty()) {
        patterns_[current_pattern.name] = current_pattern;
    }
    
    return true;
}

std::string parse_value(const std::string& line) {
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
        std::string value = line.substr(colon + 1);
        value.erase(0, value.find_first_not_of(" \t\""));
        value.erase(value.find_last_not_of(" \t\"") + 1);
        return value;
    }
    return "";
}
```

This simple parser can handle the basic YAML structure needed for patterns without external dependencies.

---

## Next Steps

1. Review and refine this plan
2. Choose YAML parsing approach (yaml-cpp vs simple parser)
3. Start with Phase 1: Create data structures and test pattern loading
4. Proceed incrementally through phases
5. Test with simple patterns before complex ones
