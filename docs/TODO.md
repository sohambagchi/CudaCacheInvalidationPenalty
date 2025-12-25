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
