#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <numa.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <thread>
#include <vector>
#include <pthread.h>
#include "pattern_config.hpp"
#include "pattern_dispatch.cuh"
#include "pattern_dispatch_cpu.hpp"

// Global pointer to active pattern for CPU threads
const PatternConfig* g_active_pattern = nullptr;

/**
 * @brief Write timing data to CSV file with timestamp
 * 
 * @param pattern_name Name of the pattern being tested
 * @param gpu_timing Host copy of GPU timing data
 * @param num_gpu_threads Total number of GPU threads
 * @param cpu_timing Host copy of CPU timing data
 * @param num_cpu_threads Total number of CPU threads
 * @param clock_overhead Average clock overhead (cycles)
 */
void write_timing_csv(const std::string& pattern_name, 
                      const gpu_timing_data* gpu_timing,
                      int num_gpu_threads,
                      const cpu_timing_data* cpu_timing,
                      int num_cpu_threads,
                      clock_t clock_overhead,
                      const bufferElement_na *gpu_results,
                      const bufferElement_na *cpu_results) {
    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = timestamp_stream.str();
    
    // Create filename
    std::string filename = "timing_" + pattern_name + "_" + timestamp + ".csv";
    
    printf("[INFO] Writing timing data to CSV file: %s\n", filename.c_str());

    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "[ERROR] Could not open timing CSV file: " << filename << std::endl;
        return;
    }
    
    // Write header
    csv << "device,block,thread,role,ordering,scope,watch_flag,start_time,flag_time,end_time,duration,wait_time,duration_corrected,buffer_read_time,per_element_read_time,caching,final_value\n";
    
    // Convert consumer_type to string
    auto consumer_type_str = [](uint8_t type) {
        switch (type) {
            case 0: return "inactive";
            case 1: return "reader_acq";
            case 2: return "reader_rlx";
            case 3: return "dummy";
            default: return "unknown";
        }
    };
    
    // Convert flag_type to string
    auto flag_type_str = [](uint8_t type) {
        switch (type) {
            case 0: return "thread";
            case 1: return "block";
            case 2: return "device";
            case 3: return "system";
            default: return "unknown";
        }
    };
    
    // Write GPU timing data
    for (int i = 0; i < num_gpu_threads; i++) {
        const auto& t = gpu_timing[i];
        
        // Skip inactive threads
        if (t.consumer_type == 0) continue;
        
        clock_t duration = t.end_time - t.start_time;
        clock_t wait_time = t.flag_trigger_time - t.start_time;
        
        // Correct for overhead (approximately)
        clock_t duration_corrected = duration > (3 * clock_overhead) ? duration - (3 * clock_overhead) : duration;
        
        // Calculate buffer read time (time between flag trigger and completion)
        clock_t buffer_read_time = t.end_time - t.flag_trigger_time;
        double per_element_read_time = static_cast<double>(buffer_read_time) / BUFFER_SIZE;
        
        // Get result value for this thread
        int tid = t.block_id * 64 + t.thread_id;
        uint32_t result_value = gpu_results[tid].data;
        
        csv << "gpu,"
            << t.block_id << ","
            << t.thread_id << ","
            << consumer_type_str(t.consumer_type) << ","
            << (t.consumer_type == 1 ? "acquire" : "relaxed") << ","
            << "N/A,"  // scope (for readers)
            << flag_type_str(t.flag_type) << ","
            << t.start_time << ","
            << t.flag_trigger_time << ","
            << t.end_time << ","
            << duration << ","
            << wait_time << ","
            << duration_corrected << ","
            << buffer_read_time << ","
            << per_element_read_time << ","
            << (t.caching ? "true" : "false") << ","
            << result_value << "\n";
    }
    
    // Write CPU timing data
    for (int i = 0; i < num_cpu_threads; i++) {
        const auto& t = cpu_timing[i];
        
        // Skip inactive threads
        if (t.consumer_type == 0) continue;
        
        uint64_t duration = t.end_ns - t.start_ns;
        uint64_t wait_time = t.flag_trigger_ns - t.start_ns;
        
        // Calculate buffer read time (time between flag trigger and completion)
        uint64_t buffer_read_time = t.end_ns - t.flag_trigger_ns;
        double per_element_read_time = static_cast<double>(buffer_read_time) / BUFFER_SIZE;
        
        uint32_t result_value = cpu_results[i].data;
        
        csv << "cpu,"
            << "N/A,"  // No blocks on CPU
            << t.thread_id << ","
            << consumer_type_str(t.consumer_type) << ","
            << (t.consumer_type == 1 ? "acquire" : "relaxed") << ","
            << "N/A,"  // scope (for readers)
            << flag_type_str(t.flag_type) << ","
            << t.start_ns << ","
            << t.flag_trigger_ns << ","
            << t.end_ns << ","
            << duration << ","
            << wait_time << ","
            << duration << ","  // No overhead correction for CPU
            << buffer_read_time << ","
            << per_element_read_time << ","
            << (t.caching ? "true" : "false") << ","
            << result_value << "\n";
    }
    
    csv.close();
    std::cout << "[INFO] Timing data written to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "[INFO] Usage: " << argv[0] << " -P <pattern_name> [-F <pattern_file>] [-m <memory_type>]" << std::endl;
        std::cout << "\n[INFO] To list available patterns: " << argv[0] << " -F <pattern_file>" << std::endl;
        return 0;
    }

    AllocatorType allocator_t = CE_SYS_MALLOC;
    std::string pattern_name = "";
    std::string pattern_file = "configs/isolated_acquire.yaml";

    int opt;
    while ((opt = getopt(argc, argv, "m:P:F:")) != -1) {
        switch (opt) {
            case 'm':
                if (strcmp(optarg, "malloc") == 0) {
                    allocator_t = CE_SYS_MALLOC;
                } else if (strcmp(optarg, "numa_host") == 0) {
                    allocator_t = CE_NUMA_HOST;
                } else if (strcmp(optarg, "numa_device") == 0) {
                    allocator_t = CE_NUMA_DEVICE;
                } else if (strcmp(optarg, "dram") == 0) {
                    allocator_t = CE_DRAM;
                } else if (strcmp(optarg, "um") == 0) {
                    allocator_t = CE_UM;
                } else if (strcmp(optarg, "cuda_malloc") == 0) {
                    allocator_t = CE_CUDA_MALLOC;
                } else {
                    std::cout << "Error: Invalid memory type" << std::endl;
                    return 0;
                }

                std::cout << "[INFO] Memory Allocator: " << optarg << std::endl;
                break;
            case 'P':
                pattern_name = optarg;
                break;
            case 'F':
                pattern_file = optarg;
                break;
            default:
                std::cout << "Error: Invalid argument" << std::endl;
                return 0;
        }
    }

    // Pattern-based mode initialization
    std::cout << "[INFO] Pattern Dispatch Mode Enabled" << std::endl;
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
    
    g_active_pattern = pattern;
    
    std::cout << "[INFO] Using pattern: " << pattern->name << std::endl;
    std::cout << "[INFO] Description: " << pattern->description << std::endl;
    
    // Copy pattern to device constant memory
    cudaMemcpyToSymbol(d_pattern_gpu, pattern->gpu_threads,
                       sizeof(ThreadConfig) * 8 * 64);

    const AllocatorType allocator = allocator_t;

    bufferElement * buffer;

    if (allocator == CE_SYS_MALLOC) {
        std::cout << "[INFO] Allocating buffer using system malloc" << std::endl;
        buffer = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));
        memset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_NUMA_HOST) {
        std::cout << "[INFO] Allocating buffer using numa_alloc_onnode (cpu)" << std::endl;
        buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 0);
        memset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_NUMA_DEVICE) {
        std::cout << "[INFO] Allocating buffer using numa_alloc_onnode (gpu)" << std::endl;
        buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 1);
        memset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_DRAM) {
        std::cout << "[INFO] Allocating buffer using cudaMallocHost" << std::endl;
        cudaMallocHost(&buffer, BUFFER_SIZE * sizeof(bufferElement));
        cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_UM) {
        std::cout << "[INFO] Allocating buffer using cudaMallocManaged" << std::endl;
        cudaMallocManaged(&buffer, BUFFER_SIZE * sizeof(bufferElement));
        cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_CUDA_MALLOC) {
        std::cout << "[INFO] Allocating buffer using cudaMalloc" << std::endl;
        cudaMalloc(&buffer, BUFFER_SIZE * sizeof(bufferElement));
        cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(bufferElement));
    }

    std::cout << "[INFO] Size of Buffer: " << sizeof(bufferElement) * BUFFER_SIZE << "B | " << sizeof(bufferElement) * BUFFER_SIZE / 1024 << "KB | " << sizeof(bufferElement) * BUFFER_SIZE / 1024 / 1024 << "MB" << std::endl;
    std::cout << "[INFO] Size of Data (in Buffer): " << sizeof(cuda::atomic<DATA_SIZE, cuda::thread_scope_system>) * BUFFER_SIZE << "B | " << sizeof(cuda::atomic<DATA_SIZE, cuda::thread_scope_system>) * BUFFER_SIZE / 1024 << "KB | " << sizeof(cuda::atomic<DATA_SIZE, cuda::thread_scope_system>) * BUFFER_SIZE / 1024 / 1024 << "MB" << std::endl;
    
    std::cout << "\n[INFO] Executing Pattern-Based Test" << std::endl;
    
    // Allocate signals and results
    flag_d *r_signal;
    flag_t *w_t_signal;
    flag_b *w_b_signal;
    flag_d *w_d_signal;
    flag_s *w_s_signal;
    flag_s *w_fb_signal;
    
    bufferElement_na *result_g;
    bufferElement *dummy_buffer;
    
    if (allocator == CE_SYS_MALLOC) {
        r_signal = (flag_d *) malloc(sizeof(flag_d));
        w_t_signal = (flag_t *) malloc(sizeof(flag_t));
        w_b_signal = (flag_b *) malloc(sizeof(flag_b));
        w_d_signal = (flag_d *) malloc(sizeof(flag_d));
        w_s_signal = (flag_s *) malloc(sizeof(flag_s));
        w_fb_signal = (flag_s *) malloc(sizeof(flag_s));
        dummy_buffer = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));
        
        memset(r_signal, 0, sizeof(flag_d));
        memset(w_t_signal, 0, sizeof(flag_t));
        memset(w_b_signal, 0, sizeof(flag_b));
        memset(w_d_signal, 0, sizeof(flag_d));
        memset(w_s_signal, 0, sizeof(flag_s));
        memset(w_fb_signal, 0, sizeof(flag_s));
        memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
    } else if (allocator == CE_DRAM) {
        cudaMallocHost(&r_signal, sizeof(flag_d));
        cudaMallocHost(&w_t_signal, sizeof(flag_t));
        cudaMallocHost(&w_b_signal, sizeof(flag_b));
        cudaMallocHost(&w_d_signal, sizeof(flag_d));
        cudaMallocHost(&w_s_signal, sizeof(flag_s));
        cudaMallocHost(&w_fb_signal, sizeof(flag_s));
        cudaMallocHost(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));
        
        cudaMemset(r_signal, 0, sizeof(flag_d));
        cudaMemset(w_t_signal, 0, sizeof(flag_t));
        cudaMemset(w_b_signal, 0, sizeof(flag_b));
        cudaMemset(w_d_signal, 0, sizeof(flag_d));
        cudaMemset(w_s_signal, 0, sizeof(flag_s));
        cudaMemset(w_fb_signal, 0, sizeof(flag_s));
        cudaMemset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
    } else {
        // NUMA or UM
        r_signal = (flag_d *) numa_alloc_onnode(sizeof(flag_d), allocator == CE_NUMA_DEVICE ? 1 : 0);
        w_t_signal = (flag_t *) numa_alloc_onnode(sizeof(flag_t), allocator == CE_NUMA_DEVICE ? 1 : 0);
        w_b_signal = (flag_b *) numa_alloc_onnode(sizeof(flag_b), allocator == CE_NUMA_DEVICE ? 1 : 0);
        w_d_signal = (flag_d *) numa_alloc_onnode(sizeof(flag_d), allocator == CE_NUMA_DEVICE ? 1 : 0);
        w_s_signal = (flag_s *) numa_alloc_onnode(sizeof(flag_s), allocator == CE_NUMA_DEVICE ? 1 : 0);
        w_fb_signal = (flag_s *) numa_alloc_onnode(sizeof(flag_s), allocator == CE_NUMA_DEVICE ? 1 : 0);
        dummy_buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), allocator == CE_NUMA_DEVICE ? 1 : 0);
        
        memset(r_signal, 0, sizeof(flag_d));
        memset(w_t_signal, 0, sizeof(flag_t));
        memset(w_b_signal, 0, sizeof(flag_b));
        memset(w_d_signal, 0, sizeof(flag_d));
        memset(w_s_signal, 0, sizeof(flag_s));
        memset(w_fb_signal, 0, sizeof(flag_s));
        memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
    }
    
    cudaMalloc(&result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
    cudaMemset(result_g, 0, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
    
    // ========================================================================
    // TIMING INSTRUMENTATION: Allocate timing arrays
    // ========================================================================
    
    std::cout << "[INFO] Allocating timing instrumentation arrays..." << std::endl;
    
    gpu_timing_data *timing_g;  // Device timing array
    cudaMalloc(&timing_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data));
    cudaMemset(timing_g, 0, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data));
    
    // Calibrate clock overhead
    std::cout << "[INFO] Calibrating clock64() overhead..." << std::endl;
    clock_t *calibration_result_d;
    cudaMalloc(&calibration_result_d, sizeof(clock_t));
    
    const int calibration_samples = 10000;
    calibrate_clock_overhead<<<1, 1>>>(calibration_result_d, calibration_samples);
    cudaDeviceSynchronize();
    
    clock_t clock_overhead_avg;
    cudaMemcpy(&clock_overhead_avg, calibration_result_d, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(calibration_result_d);
    
    std::cout << "[INFO] Average clock64() overhead: " << clock_overhead_avg << " cycles" << std::endl;
    
    // CPU timing allocation
    cpu_timing_data *timing_c = (cpu_timing_data *) malloc(CPU_NUM_THREADS * sizeof(cpu_timing_data));
    memset(timing_c, 0, CPU_NUM_THREADS * sizeof(cpu_timing_data));
    
    // CPU results allocation
    bufferElement_na *result_c = (bufferElement_na *) malloc(CPU_NUM_THREADS * sizeof(bufferElement_na));
    memset(result_c, 0, CPU_NUM_THREADS * sizeof(bufferElement_na));
    
    // ========================================================================
    
    // Check if multi-writer mode
    if (pattern->multi_writer) {
        std::cout << "[INFO] Multi-writer mode detected - allocating scope-specific buffers" << std::endl;
        
        bufferElement_t *buffer_t;
        bufferElement_b *buffer_b;
        bufferElement_d *buffer_d;
        bufferElement_s *buffer_s;
        
        if (allocator == CE_SYS_MALLOC || allocator == CE_NUMA_HOST || allocator == CE_NUMA_DEVICE) {
            buffer_t = (bufferElement_t *) malloc(BUFFER_SIZE * sizeof(bufferElement_t));
            buffer_b = (bufferElement_b *) malloc(BUFFER_SIZE * sizeof(bufferElement_b));
            buffer_d = (bufferElement_d *) malloc(BUFFER_SIZE * sizeof(bufferElement_d));
            buffer_s = (bufferElement_s *) malloc(BUFFER_SIZE * sizeof(bufferElement_s));
            memset(buffer_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));
        } else if (allocator == CE_DRAM) {
            cudaMallocHost(&buffer_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMallocHost(&buffer_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMallocHost(&buffer_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMallocHost(&buffer_s, BUFFER_SIZE * sizeof(bufferElement_s));
            cudaMemset(buffer_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMemset(buffer_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMemset(buffer_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMemset(buffer_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));
        } else if (allocator == CE_UM) {
            cudaMallocManaged(&buffer_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMallocManaged(&buffer_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMallocManaged(&buffer_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMallocManaged(&buffer_s, BUFFER_SIZE * sizeof(bufferElement_s));
            cudaMemset(buffer_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMemset(buffer_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMemset(buffer_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMemset(buffer_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));
        } else if (allocator == CE_CUDA_MALLOC) {
            cudaMalloc(&buffer_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMalloc(&buffer_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMalloc(&buffer_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMalloc(&buffer_s, BUFFER_SIZE * sizeof(bufferElement_s));
            cudaMemset(buffer_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMemset(buffer_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMemset(buffer_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMemset(buffer_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));
        }
        
        std::cout << "[INFO] Launching CPU threads for multi-writer mode..." << std::endl;
        std::vector<std::thread> cpu_threads;
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads.push_back(std::thread(dispatch_cpu_thread_multi, i,
                buffer_t, buffer_b, buffer_d, buffer_s, dummy_buffer, result_c,
                r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                w_fb_signal, timing_c));
            
            // Set CPU affinity
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_setaffinity_np(cpu_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        
        std::cout << "[INFO] Launching pattern_orchestrator_multi kernel..." << std::endl;
        
        pattern_orchestrator_multi<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(
            buffer_t, buffer_b, buffer_d, buffer_s,
            dummy_buffer, result_g,
            r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
            w_fb_signal, timing_g
        );
        
        // Wait for CPU threads to complete
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads[i].join();
        }
        
        // Cleanup multi-writer buffers
        if (allocator == CE_SYS_MALLOC || allocator == CE_NUMA_HOST || allocator == CE_NUMA_DEVICE) {
            free(buffer_t);
            free(buffer_b);
            free(buffer_d);
            free(buffer_s);
        } else if (allocator == CE_DRAM) {
            cudaFreeHost(buffer_t);
            cudaFreeHost(buffer_b);
            cudaFreeHost(buffer_d);
            cudaFreeHost(buffer_s);
        } else if (allocator == CE_UM || allocator == CE_CUDA_MALLOC) {
            cudaFree(buffer_t);
            cudaFree(buffer_b);
            cudaFree(buffer_d);
            cudaFree(buffer_s);
        }
    } else {
        std::cout << "[INFO] Launching CPU threads for single-writer mode..." << std::endl;
        std::vector<std::thread> cpu_threads;
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads.push_back(std::thread(dispatch_cpu_thread, i,
                buffer, dummy_buffer, result_c,
                r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
                w_fb_signal, timing_c));
            
            // Set CPU affinity
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_setaffinity_np(cpu_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        
        std::cout << "[INFO] Launching pattern_orchestrator kernel..." << std::endl;
        
        // Launch pattern orchestrator
        pattern_orchestrator<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(
            buffer, dummy_buffer, result_g,
            r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal,
            w_fb_signal, timing_g
        );
        
        // Wait for CPU threads to complete
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads[i].join();
        }
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Kernel execution completed" << std::endl;
    
    // ========================================================================
    // TIMING INSTRUMENTATION: Copy and write timing data
    // ========================================================================
    
    std::cout << "[INFO] Copying timing data from device..." << std::endl;
    gpu_timing_data *timing_h = (gpu_timing_data *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data));
    cudaMemcpy(timing_h, timing_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(gpu_timing_data), cudaMemcpyDeviceToHost);
    
    // Copy results back
    bufferElement_na *result_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
    cudaMemcpy(result_h, result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);
    
    // Write timing CSV (includes both GPU and CPU)
    write_timing_csv(pattern->name, timing_h, GPU_NUM_BLOCKS * GPU_NUM_THREADS, 
                     timing_c, CPU_NUM_THREADS, clock_overhead_avg,
                     result_h, result_c);
    
    // ========================================================================
    
    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    for (int i = 0; i < GPU_NUM_BLOCKS; i++) {
        for (int j = 0; j < GPU_NUM_THREADS; j++) {
            int idx = i * GPU_NUM_THREADS + j;
            ThreadConfig cfg = pattern->gpu_threads[i][j];
            if (cfg.role != ThreadRole::INACTIVE && cfg.role != ThreadRole::DUMMY_READER && cfg.role != ThreadRole::DUMMY_WRITER) {
                std::cout << "B[" << i << "] T[" << j << "] " 
                          << role_to_string(cfg.role) << " "
                          << ordering_to_string(cfg.ordering);
                
                // For dummy threads, don't print the garbage value
                if (cfg.role == ThreadRole::DUMMY_READER || cfg.role == ThreadRole::DUMMY_WRITER) {
                    std::cout << " (dummy)" << std::endl;
                } else {
                    std::cout << " Result: " << result_h[idx].data << std::endl;
                }
            }
        }
    }
    
    // Print CPU results
    std::cout << "\n=== CPU Results ===" << std::endl;
    for (int i = 0; i < CPU_NUM_THREADS; i++) {
        ThreadConfig cfg = pattern->cpu_threads[i];
        if (cfg.role != ThreadRole::INACTIVE) {
            std::cout << "C[" << i << "] " 
                      << role_to_string(cfg.role) << " "
                      << ordering_to_string(cfg.ordering)
                      << " Result: " << result_c[i].data << std::endl;
        }
    }
    
    // Cleanup
    free(result_h);
    free(result_c);
    free(timing_h);
    free(timing_c);
    cudaFree(result_g);
    cudaFree(timing_g);
    
    if (allocator == CE_SYS_MALLOC) {
        free(r_signal);
        free(w_t_signal);
        free(w_b_signal);
        free(w_d_signal);
        free(w_s_signal);
        free(w_fb_signal);
        free(dummy_buffer);
    } else if (allocator == CE_DRAM) {
        cudaFreeHost(r_signal);
        cudaFreeHost(w_t_signal);
        cudaFreeHost(w_b_signal);
        cudaFreeHost(w_d_signal);
        cudaFreeHost(w_s_signal);
        cudaFreeHost(w_fb_signal);
        cudaFreeHost(dummy_buffer);
    } else {
        numa_free(r_signal, sizeof(flag_d));
        numa_free(w_t_signal, sizeof(flag_t));
        numa_free(w_b_signal, sizeof(flag_b));
        numa_free(w_d_signal, sizeof(flag_d));
        numa_free(w_s_signal, sizeof(flag_s));
        numa_free(w_fb_signal, sizeof(flag_s));
        numa_free(dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));
    }
    
    if (allocator == CE_SYS_MALLOC || allocator == CE_NUMA_HOST || allocator == CE_NUMA_DEVICE) {
        free(buffer);
    } else if (allocator == CE_DRAM) {
        cudaFreeHost(buffer);
    } else if (allocator == CE_UM || allocator == CE_CUDA_MALLOC) {
        cudaFree(buffer);
    }
    
    std::cout << "\n[INFO] Pattern test completed successfully" << std::endl;
    return 0;
}
