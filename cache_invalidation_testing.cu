// #include "cache_invalidation_testing_propagation_hierarchy.cu"
#include "cache_invalidation_testing.cuh"

int main(int argc, char* argv[]) {

    if (argc < 6) {
        std::cout << "Error: Insufficient arguments" << std::endl; 

        // print all the arguments
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << std::endl;
        }

        // generate usage message
        std::cout << "Usage: " << argv[0] << " -m <memory_type> -r <reader_type> -w <writer_type> -o <output_file>" << std::endl;
        return 0;
    }

    AllocatorType allocator_t;
    ReaderWriterType reader_t;
    ReaderWriterType writer_t;

    bool multi_producer = false;

    int opt;

    while ((opt = getopt(argc, argv, "m:r:w:p")) != -1) {
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
            case 'r':
                if (strcmp(optarg, "gpu") == 0) {
                    reader_t = CE_GPU;
                } else if (strcmp(optarg, "cpu") == 0) {
                    reader_t = CE_CPU;
                } else {
                    std::cout << "Error: Invalid reader type" << std::endl;
                    return 0;
                }
            case 'w':
                if (strcmp(optarg, "gpu") == 0) {
                    writer_t = CE_GPU;
                } else if (strcmp(optarg, "cpu") == 0) {
                    writer_t = CE_CPU;
                } else {
                    std::cout << "Error: Invalid writer type" << std::endl;
                    return 0;
                }
            case 'p':
                multi_producer = true;
                break;
            default:
                std::cout << "Error: Invalid argument" << std::endl;
                return 0;
           
        }
    }

    const AllocatorType allocator = allocator_t;
    const ReaderWriterType reader = reader_t;
    const ReaderWriterType writer = writer_t;

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
    std::cout << "[INFO] Size of Data (in Buffer): " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE << "B | " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE / 1024 << "KB | " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE / 1024 / 1024 << "MB" << std::endl;
    
    if (reader == CE_GPU) {
        std::cout << "[INFO] Reader: GPU" << std::endl;
    } else if (reader == CE_CPU) {
        std::cout << "[INFO] Reader: CPU" << std::endl;
    }

    if (writer == CE_GPU) {
        std::cout << "[INFO] Writer: GPU" << std::endl;
    } else if (writer == CE_CPU) {
        std::cout << "[INFO] Writer: CPU" << std::endl;
    }

    
    /**
      * SOBA_COMMENT: CPU GPU Time Scales
      + - CPU measured using std::chrono
      * - GPU measured using clock64
      **/

    // if (reader == CE_CPU && writer == CE_CPU) {
    //     gpu_reader_cycles *= 1000;
    //     cpu_reader_ns *= 1000;
    //     writer_invocation_delay = 250;
    // } else if (reader == CE_GPU && writer == CE_GPU) {
    //     gpu_reader_cycles *= 10000000;
    //     cpu_reader_ns *= 10000000;
    //     writer_invocation_delay = 10;
    // } else if (reader == CE_CPU && writer == CE_GPU) {
    //     gpu_reader_cycles *= 100;
        // cpu_reader_ns *= 100;
    //     writer_invocation_delay = 1000;
    // } else if (reader == CE_GPU && writer == CE_CPU) {
    //     gpu_reader_cycles *= 50;
    //     cpu_reader_ns *= 50;
    //     writer_invocation_delay = 1000;
    // }

    // if (writer == CE_GPU) {
    //     gpu_reader_cycles *= 100;
    //     cpu_reader_ns *= 100;
    // } else {
    //     gpu_reader_cycles *= 100;
    //     cpu_reader_ns *= 100;
    // }
    // gpu_reader_cycles *= 10;
    // cpu_reader_ns *= 10;
    
    cpu_set_t cpuset;

    if (reader == CE_GPU && writer == CE_GPU) {
        std::cout << "[INFO] Spawning GPU Reader and Writer" << std::endl;

        bufferElement *buffer_g;
        cudaMalloc(&buffer_g, BUFFER_SIZE * sizeof(bufferElement));

        cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *g_r_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_thread> *g_w_t_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_block> *g_w_b_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_device> *g_w_d_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *g_w_s_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *g_w_fb_signal; // Fallback Signal

        WriterType *g_spawn_writer;
        WriterType c_spawn_writer = CE_WRITER;

        cudaMalloc(&g_r_signal, 1 * sizeof(cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE>));
        cudaMalloc(&g_w_t_signal, 1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_thread>));
        cudaMalloc(&g_w_b_signal, 1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_block>));
        cudaMalloc(&g_w_d_signal, 1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>));
        cudaMalloc(&g_w_s_signal, 1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));
        cudaMalloc(&g_w_fb_signal, 1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));

        cudaMalloc(&g_spawn_writer, 1 * sizeof(WriterType));
        cudaMemcpy(g_spawn_writer, &c_spawn_writer, 1 * sizeof(WriterType), cudaMemcpyHostToDevice);


        bufferElement_na * result;
        cudaMalloc(&result, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

        gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(buffer, buffer_g, result, g_r_signal, g_w_t_signal, g_w_b_signal, g_w_d_signal, g_w_s_signal, g_w_fb_signal, g_spawn_writer);

        cudaDeviceSynchronize();

        bufferElement_na * result_h;
        result_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

        cudaMemcpy(result_h, result, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);

        for (int i = 0; i < GPU_NUM_BLOCKS; i++) {
            for (int j = 0; j < GPU_NUM_THREADS; j++) {
                if (result_h[i * GPU_NUM_THREADS + j].data > 2000000000) {
                    std::cout << "[" << i << "][" << j << "] " << "--" << std::endl;
                } else {
                    std::cout << "[" << i << "][" << j << "] " << result_h[i * GPU_NUM_THREADS + j].data << std::endl;
                }
            }
        }

    } else if (reader == CE_CPU && writer == CE_CPU) {

        std::cout << "[INFO] Spawning CPU Reader and Writer" << std::endl;

        bufferElement *buffer_c;

        buffer_c = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));

        cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *c_r_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_thread> *c_w_t_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_block> *c_w_b_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_device> *c_w_d_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *c_w_s_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *c_w_fb_signal;

        WriterType spawn_writer = CE_WRITER;

        c_r_signal = (cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *) malloc(1 * sizeof(cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE>));
        c_w_t_signal = (cuda::atomic<uint32_t, cuda::thread_scope_thread> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_thread>));
        c_w_b_signal = (cuda::atomic<uint32_t, cuda::thread_scope_block> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_block>));
        c_w_d_signal = (cuda::atomic<uint32_t, cuda::thread_scope_device> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>));
        c_w_s_signal = (cuda::atomic<uint32_t, cuda::thread_scope_system> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));
        c_w_fb_signal = (cuda::atomic<uint32_t, cuda::thread_scope_system> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));

        c_r_signal->store(0);
        c_w_t_signal->store(0);
        c_w_b_signal->store(0);
        c_w_d_signal->store(0);
        c_w_s_signal->store(0);
        c_w_fb_signal->store(0);

        bufferElement_na *result;
        result = (bufferElement_na *) malloc(CPU_NUM_THREADS * sizeof(bufferElement_na));

        std::vector<std::thread> cpu_threads;

        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            result[i].data = 0;
            cpu_threads.push_back(std::thread(cpu_buffer_reader_writer_propagation_hierarchy, buffer, buffer_c, result, c_r_signal, c_w_t_signal, c_w_b_signal, c_w_d_signal, c_w_s_signal, c_w_fb_signal, &spawn_writer));

            CPU_ZERO(&cpuset);
            if (i == 0) {
                CPU_SET(i, &cpuset);
            } else {
                CPU_SET(i+32, &cpuset);
            }
            pthread_setaffinity_np(cpu_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads[i].join();
        }

        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            std::cout << "C[" << i << "]";
            
            if (i % 4 == 0) {
                std::cout << " Thread";
            } else if (i % 4 == 1) {
                std::cout << " Block";
            } else if (i % 4 == 2) {
                std::cout << " Device";
            } else {
                std::cout << " System";
            }
            
            if (i % 8 < 4) {
                std::cout << "-Rlx ";
            } else {
                std::cout << "-Acq ";
            }
            
            std::cout << result[i].data << std::endl;
        }
    } else {
        
        bufferElement * dummy_buffer;
        dummy_buffer = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));
        
        bufferElement_na * result_g;
        bufferElement_na * result_c;
        
        cudaMalloc(&result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
        result_c = (bufferElement_na *) malloc(CPU_NUM_THREADS * sizeof(bufferElement_na));
        
        WriterType *g_writer_type;
        WriterType h_writer_type;
        WriterType *c_writer_type;
        
        cudaMalloc(&g_writer_type, 1 * sizeof(WriterType));
        c_writer_type = (WriterType *) malloc(1 * sizeof(WriterType));
        
        if (writer == CE_GPU) {
            h_writer_type = CE_HET_WRITER;
            *c_writer_type = CE_NO_WRITER;
        } else {
            h_writer_type = CE_NO_WRITER;
            *c_writer_type = CE_HET_WRITER;
        }
        
        cudaMemcpy(g_writer_type, &h_writer_type, 1 * sizeof(WriterType), cudaMemcpyHostToDevice);
        
        cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *cg_r_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_thread> *cg_w_t_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_block> *cg_w_b_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_device> *cg_w_d_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *cg_w_s_signal;
        cuda::atomic<uint32_t, cuda::thread_scope_system> *cg_w_fb_signal;
        
        cg_r_signal = (cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE> *) malloc(1 * sizeof(cuda::atomic<uint32_t, SIGNAL_THREAD_SCOPE>));
        cg_w_t_signal = (cuda::atomic<uint32_t, cuda::thread_scope_thread> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_thread>));
        cg_w_b_signal = (cuda::atomic<uint32_t, cuda::thread_scope_block> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_block>));
        cg_w_d_signal = (cuda::atomic<uint32_t, cuda::thread_scope_device> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>));
        cg_w_s_signal = (cuda::atomic<uint32_t, cuda::thread_scope_system> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));
        cg_w_fb_signal = (cuda::atomic<uint32_t, cuda::thread_scope_system> *) malloc(1 * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_system>));
        
        cg_r_signal->store(0);
        cg_w_t_signal->store(0);
        cg_w_b_signal->store(0);
        cg_w_d_signal->store(0);
        cg_w_s_signal->store(0);
        cg_w_fb_signal->store(0);
        
        std::cout << "[INFO] Spawning Heterogeneous Reader and Writer" << std::endl;
        
        gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(buffer, dummy_buffer, result_g, cg_r_signal, cg_w_t_signal, cg_w_b_signal, cg_w_d_signal, cg_w_s_signal, cg_w_fb_signal, g_writer_type);

        std::vector<std::thread> cpu_threads;
        
        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            result_c[i].data = 0;
            cpu_threads.push_back(std::thread(cpu_buffer_reader_writer_propagation_hierarchy, buffer, dummy_buffer, result_c, cg_r_signal, cg_w_t_signal, cg_w_b_signal, cg_w_d_signal, cg_w_s_signal, cg_w_fb_signal, c_writer_type));
            
            CPU_ZERO(&cpuset);
            if (i == 0) {
                CPU_SET(i, &cpuset);
            } else {
                CPU_SET(i+32, &cpuset);
            }
            // CPU_SET(i, &cpuset);
            pthread_setaffinity_np(cpu_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            cpu_threads[i].join();
        }

        std::cout << "[INFO] GPU Results" << std::endl;

        bufferElement_na * result_g_h;
        result_g_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

        cudaMemcpy(result_g_h, result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);

        for (int i = 0; i < GPU_NUM_BLOCKS; i++) {
            for (int j = 0; j < GPU_NUM_THREADS; j++) {
                if (result_g_h[i * GPU_NUM_THREADS + j].data > 2000000000) {
                    std::cout << "[" << i << "][" << j << "] " << "--" << std::endl;
                } else {
                    std::cout << "[" << i << "][" << j << "] " << result_g_h[i * GPU_NUM_THREADS + j].data << std::endl;
                }
            }
        }

        std::cout << "[INFO] CPU Results" << std::endl;

        for (int i = 0; i < CPU_NUM_THREADS; i++) {
            std::cout << "C[" << i << "]";
            
            if (i % 4 == 0) {
                std::cout << " Thread";
            } else if (i % 4 == 1) {
                std::cout << " Block";
            } else if (i % 4 == 2) {
                std::cout << " Device";
            } else {
                std::cout << " System";
            }

            if (i % 8 < 4) {
                std::cout << "-Rlx ";
            } else {
                std::cout << "-Acq ";
            }

            std::cout << result_c[i].data << std::endl;
        }
        
    }

    std::cout << "[INFO] Done, Freeing Memory" << std::endl;
    

    if (allocator == CE_SYS_MALLOC) {
        free(buffer);
    } else if (allocator == CE_NUMA_HOST || allocator == CE_NUMA_DEVICE) {
        numa_free(buffer, BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>));
    } else if (allocator == CE_DRAM) {
        cudaFreeHost(buffer);
    } else if (allocator == CE_UM) {
        cudaFree(buffer);
    } else if (allocator == CE_CUDA_MALLOC) {
        cudaFree(buffer);
    }
    
    return 0;
}
