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
                break;
            case 'w':
                if (strcmp(optarg, "gpu") == 0) {
                    writer_t = CE_GPU;
                } else if (strcmp(optarg, "cpu") == 0) {
                    writer_t = CE_CPU;
                } else {
                    std::cout << "Error: Invalid writer type" << std::endl;
                    return 0;
                }
                break;
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
        if (reader == CE_CPU || writer == CE_CPU) {
            std::cout << "[ERROR] CPU cannot read/write from/to GPU buffer" << std::endl;
            return 0;
        }
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

    if (multi_producer)
        std::cout << "[INFO] Multi-Producer Mode" << std::endl;


    #ifdef NO_ACQ
        std::cout << "[INFO] No Acquire" << std::endl;
    #endif

    #ifdef P_H_FLAG_STORE_ORDER_REL
        std::cout << "[INFO] Producers Release-Store to Flags" << std::endl;
    #endif

    #ifdef P_H_FLAG_STORE_ORDER_RLX
        std::cout << "[INFO] Producers Relaxed-Store to Flags" << std::endl;
    #endif

    
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

    if (multi_producer) {
        bufferElement_t * buffer_g_t; // GPU Thread Scope
        bufferElement_b * buffer_g_b; // GPU Block Scope
        bufferElement_d * buffer_g_d; // GPU Device Scope
        bufferElement_s * buffer_g_s; // GPU System Scope
        
        flag_d *r_signal; // Cache-Ready Signal
        flag_t *w_signal_t;
        flag_b *w_signal_b;
        flag_d *w_signal_d;
        flag_s *w_signal_s;
        flag_s *w_signal_fb; // Fallback Signal

        bufferElement_na * result_g;
        bufferElement_na * result_c;

        bufferElement * dummy_buffer;

        if (allocator == CE_SYS_MALLOC) {
            dummy_buffer = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));

            buffer_g_t = (bufferElement_t *) malloc(BUFFER_SIZE * sizeof(bufferElement_t));
            buffer_g_b = (bufferElement_b *) malloc(BUFFER_SIZE * sizeof(bufferElement_b));
            buffer_g_d = (bufferElement_d *) malloc(BUFFER_SIZE * sizeof(bufferElement_d));
            buffer_g_s = (bufferElement_s *) malloc(BUFFER_SIZE * sizeof(bufferElement_s));
            
            r_signal = (flag_d *) malloc(1 * sizeof(flag_d));
            w_signal_t = (flag_t *) malloc(1 * sizeof(flag_t));
            w_signal_b = (flag_b *) malloc(1 * sizeof(flag_b));
            w_signal_d = (flag_d *) malloc(1 * sizeof(flag_d));
            w_signal_s = (flag_s *) malloc(1 * sizeof(flag_s));
            w_signal_fb = (flag_s *) malloc(1 * sizeof(flag_s));
                        
            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_signal_t, 0, 1 * sizeof(flag_t));
            memset(w_signal_b, 0, 1 * sizeof(flag_b));
            memset(w_signal_d, 0, 1 * sizeof(flag_d));
            memset(w_signal_s, 0, 1 * sizeof(flag_s));
            memset(w_signal_fb, 0, 1 * sizeof(flag_s));

            memset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_CUDA_MALLOC) {
            if (reader == CE_CPU || writer == CE_CPU) {
                std::cout << "[ERROR] CPU cannot read/write from/to GPU buffer" << std::endl;
                return 0;
            }
            cudaMalloc(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMalloc(&buffer_g_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMalloc(&buffer_g_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMalloc(&buffer_g_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMalloc(&buffer_g_s, BUFFER_SIZE * sizeof(bufferElement_s));
            
            cudaMalloc(&r_signal, 1 * sizeof(flag_d));
            cudaMalloc(&w_signal_t, 1 * sizeof(flag_t));
            cudaMalloc(&w_signal_b, 1 * sizeof(flag_b));
            cudaMalloc(&w_signal_d, 1 * sizeof(flag_d));
            cudaMalloc(&w_signal_s, 1 * sizeof(flag_s));
            cudaMalloc(&w_signal_fb, 1 * sizeof(flag_s));

            cudaMemset(r_signal, 0, 1 * sizeof(flag_d));
            cudaMemset(w_signal_t, 0, 1 * sizeof(flag_t));
            cudaMemset(w_signal_b, 0, 1 * sizeof(flag_b));
            cudaMemset(w_signal_d, 0, 1 * sizeof(flag_d));
            cudaMemset(w_signal_s, 0, 1 * sizeof(flag_s));
            cudaMemset(w_signal_fb, 0, 1 * sizeof(flag_s));

            cudaMemset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMemset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMemset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMemset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            cudaMemset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_NUMA_HOST) {
            dummy_buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 0);

            buffer_g_t = (bufferElement_t *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_t), 0);
            buffer_g_b = (bufferElement_b *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_b), 0);
            buffer_g_d = (bufferElement_d *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_d), 0);
            buffer_g_s = (bufferElement_s *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_s), 0);
            
            r_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 0);
            w_signal_t = (flag_t *) numa_alloc_onnode(1 * sizeof(flag_t), 0);
            w_signal_b = (flag_b *) numa_alloc_onnode(1 * sizeof(flag_b), 0);
            w_signal_d = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 0);
            w_signal_s = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 0);
            w_signal_fb = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 0);
                        
            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_signal_t, 0, 1 * sizeof(flag_t));
            memset(w_signal_b, 0, 1 * sizeof(flag_b));
            memset(w_signal_d, 0, 1 * sizeof(flag_d));
            memset(w_signal_s, 0, 1 * sizeof(flag_s));
            memset(w_signal_fb, 0, 1 * sizeof(flag_s));

            memset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_NUMA_DEVICE) {
            dummy_buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 1);

            buffer_g_t = (bufferElement_t *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_t), 1);
            buffer_g_b = (bufferElement_b *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_b), 1);
            buffer_g_d = (bufferElement_d *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_d), 1);
            buffer_g_s = (bufferElement_s *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement_s), 1);
            
            r_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 1);
            w_signal_t = (flag_t *) numa_alloc_onnode(1 * sizeof(flag_t), 1);
            w_signal_b = (flag_b *) numa_alloc_onnode(1 * sizeof(flag_b), 1);
            w_signal_d = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 1);
            w_signal_s = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 1);
            w_signal_fb = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 1);
            
            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_signal_t, 0, 1 * sizeof(flag_t));
            memset(w_signal_b, 0, 1 * sizeof(flag_b));
            memset(w_signal_d, 0, 1 * sizeof(flag_d));
            memset(w_signal_s, 0, 1 * sizeof(flag_s));
            memset(w_signal_fb, 0, 1 * sizeof(flag_s));
            
            memset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_DRAM) {
            cudaMallocHost(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMallocHost(&buffer_g_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMallocHost(&buffer_g_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMallocHost(&buffer_g_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMallocHost(&buffer_g_s, BUFFER_SIZE * sizeof(bufferElement_s));
            
            cudaMallocHost(&r_signal, 1 * sizeof(flag_d));
            cudaMallocHost(&w_signal_t, 1 * sizeof(flag_t));
            cudaMallocHost(&w_signal_b, 1 * sizeof(flag_b));
            cudaMallocHost(&w_signal_d, 1 * sizeof(flag_d));
            cudaMallocHost(&w_signal_s, 1 * sizeof(flag_s));
            cudaMallocHost(&w_signal_fb, 1 * sizeof(flag_s));
           
            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_signal_t, 0, 1 * sizeof(flag_t));
            memset(w_signal_b, 0, 1 * sizeof(flag_b));
            memset(w_signal_d, 0, 1 * sizeof(flag_d));
            memset(w_signal_s, 0, 1 * sizeof(flag_s));
            memset(w_signal_fb, 0, 1 * sizeof(flag_s));
            
            memset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_UM) {
            cudaMallocManaged(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMallocManaged(&buffer_g_t, BUFFER_SIZE * sizeof(bufferElement_t));
            cudaMallocManaged(&buffer_g_b, BUFFER_SIZE * sizeof(bufferElement_b));
            cudaMallocManaged(&buffer_g_d, BUFFER_SIZE * sizeof(bufferElement_d));
            cudaMallocManaged(&buffer_g_s, BUFFER_SIZE * sizeof(bufferElement_s));
            
            cudaMallocManaged(&r_signal, 1 * sizeof(flag_d));
            cudaMallocManaged(&w_signal_t, 1 * sizeof(flag_t));
            cudaMallocManaged(&w_signal_b, 1 * sizeof(flag_b));
            cudaMallocManaged(&w_signal_d, 1 * sizeof(flag_d));
            cudaMallocManaged(&w_signal_s, 1 * sizeof(flag_s));
            cudaMallocManaged(&w_signal_fb, 1 * sizeof(flag_s));

            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_signal_t, 0, 1 * sizeof(flag_t));
            memset(w_signal_b, 0, 1 * sizeof(flag_b));
            memset(w_signal_d, 0, 1 * sizeof(flag_d));
            memset(w_signal_s, 0, 1 * sizeof(flag_s));
            memset(w_signal_fb, 0, 1 * sizeof(flag_s));
            
            memset(buffer_g_t, 0, BUFFER_SIZE * sizeof(bufferElement_t));
            memset(buffer_g_b, 0, BUFFER_SIZE * sizeof(bufferElement_b));
            memset(buffer_g_d, 0, BUFFER_SIZE * sizeof(bufferElement_d));
            memset(buffer_g_s, 0, BUFFER_SIZE * sizeof(bufferElement_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        }

        result_c = (bufferElement_na *) malloc(CPU_NUM_THREADS * sizeof(bufferElement_na));
        cudaMalloc(&result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

        memset(result_c, 0, CPU_NUM_THREADS * sizeof(bufferElement_na));
        cudaMemset(result_g, 0, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
        
        if (reader == CE_GPU && writer == CE_GPU) {

            WriterType * spawn_writer;
            WriterType c_spawn_writer = CE_MULTI_WRITER;

            cudaMalloc(&spawn_writer, 1 * sizeof(WriterType));
            cudaMemcpy(spawn_writer, &c_spawn_writer, 1 * sizeof(WriterType), cudaMemcpyHostToDevice);

            gpu_buffer_reader_multi_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(dummy_buffer, buffer_g_t, buffer_g_b, buffer_g_d, buffer_g_s, result_g, r_signal, w_signal_t, w_signal_b, w_signal_d, w_signal_s, w_signal_fb, spawn_writer);
            
            cudaDeviceSynchronize();

            bufferElement_na * result_h;
            result_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

            cudaMemcpy(result_h, result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);

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

            WriterType spawn_writer = CE_MULTI_WRITER;

            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                result_c[i].data = 0;
            }

            std::vector<std::thread> cpu_threads;

            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                cpu_threads.push_back(std::thread(cpu_buffer_reader_multi_writer_propagation_hierarchy, dummy_buffer, buffer_g_t, buffer_g_b, buffer_g_d, buffer_g_s, result_c, r_signal, w_signal_t, w_signal_b, w_signal_d, w_signal_s, w_signal_fb, &spawn_writer));

                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
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
                
                std::cout << result_c[i].data << std::endl;
            }

        } else {

            WriterType *g_writer_type;
            WriterType h_writer_type;

            cudaMalloc(&g_writer_type, 1 * sizeof(WriterType));
            WriterType *c_writer_type;
            c_writer_type = (WriterType *) malloc(1 * sizeof(WriterType));

            if (writer == CE_GPU) {
                h_writer_type = CE_HET_WRITER;
                *c_writer_type = CE_NO_WRITER;
            } else {
                h_writer_type = CE_NO_WRITER;
                *c_writer_type = CE_HET_WRITER;
            }

            cudaMemcpy(g_writer_type, &h_writer_type, 1 * sizeof(WriterType), cudaMemcpyHostToDevice);

            std::vector<std::thread> cpu_threads;

            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                cpu_threads.push_back(std::thread(cpu_buffer_reader_multi_writer_propagation_hierarchy, dummy_buffer, buffer_g_t, buffer_g_b, buffer_g_d, buffer_g_s, result_c, r_signal, w_signal_t, w_signal_b, w_signal_d, w_signal_s, w_signal_fb, c_writer_type));

                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
                pthread_setaffinity_np(cpu_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            }

            gpu_buffer_reader_multi_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(dummy_buffer, buffer_g_t, buffer_g_b, buffer_g_d, buffer_g_s, result_g, r_signal, w_signal_t, w_signal_b, w_signal_d, w_signal_s, w_signal_fb, g_writer_type);
            
            cudaDeviceSynchronize();

            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                cpu_threads[i].join();
            }

            bufferElement_na * result_h;
            result_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

            cudaMemcpy(result_h, result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);

            for (int i = 0; i < GPU_NUM_BLOCKS; i++) {
                for (int j = 0; j < GPU_NUM_THREADS; j++) {
                    if (result_h[i * GPU_NUM_THREADS + j].data > 2000000000) {
                        std::cout << "[" << i << "][" << j << "] " << "--" << std::endl;
                    } else {
                        std::cout << "[" << i << "][" << j << "] " << result_h[i * GPU_NUM_THREADS + j].data << std::endl;
                    }
                }
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
                
                std::cout << result_c[i].data << std::endl;
            }
        }

        if (allocator == CE_SYS_MALLOC) {
            free(dummy_buffer);

            free(buffer_g_t);
            free(buffer_g_b);
            free(buffer_g_d);
            free(buffer_g_s);
            
            free(r_signal);
            free(w_signal_t);
            free(w_signal_b);
            free(w_signal_d);
            free(w_signal_s);
            free(w_signal_fb);
        } else if (allocator == CE_CUDA_MALLOC || allocator == CE_UM) {
            cudaFree(dummy_buffer);

            cudaFree(buffer_g_t);
            cudaFree(buffer_g_b);
            cudaFree(buffer_g_d);
            cudaFree(buffer_g_s);
            
            cudaFree(r_signal);
            cudaFree(w_signal_t);
            cudaFree(w_signal_b);
            cudaFree(w_signal_d);
            cudaFree(w_signal_s);
            cudaFree(w_signal_fb);
        } else if (allocator == CE_NUMA_DEVICE || allocator == CE_NUMA_HOST) {
            numa_free(dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            numa_free(buffer_g_t, BUFFER_SIZE * sizeof(bufferElement_t));
            numa_free(buffer_g_b, BUFFER_SIZE * sizeof(bufferElement_b));
            numa_free(buffer_g_d, BUFFER_SIZE * sizeof(bufferElement_d));
            numa_free(buffer_g_s, BUFFER_SIZE * sizeof(bufferElement_s));
            
            numa_free(r_signal, 1 * sizeof(flag_d));
            numa_free(w_signal_t, 1 * sizeof(flag_t));
            numa_free(w_signal_b, 1 * sizeof(flag_b));
            numa_free(w_signal_d, 1 * sizeof(flag_d));
            numa_free(w_signal_s, 1 * sizeof(flag_s));
            numa_free(w_signal_fb, 1 * sizeof(flag_s));
        } else if (allocator == CE_DRAM) {
            cudaFreeHost(dummy_buffer);

            cudaFreeHost(buffer_g_t);
            cudaFreeHost(buffer_g_b);
            cudaFreeHost(buffer_g_d);
            cudaFreeHost(buffer_g_s);
            
            cudaFreeHost(r_signal);
            cudaFreeHost(w_signal_t);
            cudaFreeHost(w_signal_b);
            cudaFreeHost(w_signal_d);
            cudaFreeHost(w_signal_s);
            cudaFreeHost(w_signal_fb);
        } 

        free(result_c);
        cudaFree(result_g);
        
    } else {

        flag_d *r_signal;
        flag_t *w_t_signal;
        flag_b *w_b_signal;
        flag_d *w_d_signal;
        flag_s *w_s_signal;
        flag_s *w_fb_signal; // Fallback Signal

        bufferElement *dummy_buffer;

        if (allocator == CE_SYS_MALLOC) {
            dummy_buffer = (bufferElement *) malloc(BUFFER_SIZE * sizeof(bufferElement));

            r_signal = (flag_d *) malloc(1 * sizeof(flag_d));
            w_t_signal = (flag_t *) malloc(1 * sizeof(flag_t));
            w_b_signal = (flag_b *) malloc(1 * sizeof(flag_b));
            w_d_signal = (flag_d *) malloc(1 * sizeof(flag_d));
            w_s_signal = (flag_s *) malloc(1 * sizeof(flag_s));
            w_fb_signal = (flag_s *) malloc(1 * sizeof(flag_s));
            
            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_t_signal, 0, 1 * sizeof(flag_t));
            memset(w_b_signal, 0, 1 * sizeof(flag_b));
            memset(w_d_signal, 0, 1 * sizeof(flag_d));
            memset(w_s_signal, 0, 1 * sizeof(flag_s));
            memset(w_fb_signal, 0, 1 * sizeof(flag_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_CUDA_MALLOC) {
            if (reader == CE_CPU || writer == CE_CPU) {
                std::cout << "[ERROR] CPU cannot read/write from/to GPU buffer" << std::endl;
                return 0;
            }
            cudaMalloc(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMalloc(&r_signal, 1 * sizeof(flag_d));
            cudaMalloc(&w_t_signal, 1 * sizeof(flag_t));
            cudaMalloc(&w_b_signal, 1 * sizeof(flag_b));
            cudaMalloc(&w_d_signal, 1 * sizeof(flag_d));
            cudaMalloc(&w_s_signal, 1 * sizeof(flag_s));
            cudaMalloc(&w_fb_signal, 1 * sizeof(flag_s));

            cudaMemset(r_signal, 0, 1 * sizeof(flag_d));
            cudaMemset(w_t_signal, 0, 1 * sizeof(flag_t));
            cudaMemset(w_b_signal, 0, 1 * sizeof(flag_b));
            cudaMemset(w_d_signal, 0, 1 * sizeof(flag_d));
            cudaMemset(w_s_signal, 0, 1 * sizeof(flag_s));
            cudaMemset(w_fb_signal, 0, 1 * sizeof(flag_s));

            cudaMemset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_NUMA_HOST) {
            dummy_buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 0);

            r_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 0);
            w_t_signal = (flag_t *) numa_alloc_onnode(1 * sizeof(flag_t), 0);
            w_b_signal = (flag_b *) numa_alloc_onnode(1 * sizeof(flag_b), 0);
            w_d_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 0);
            w_s_signal = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 0);
            w_fb_signal = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 0);

            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_t_signal, 0, 1 * sizeof(flag_t));
            memset(w_b_signal, 0, 1 * sizeof(flag_b));
            memset(w_d_signal, 0, 1 * sizeof(flag_d));
            memset(w_s_signal, 0, 1 * sizeof(flag_s));
            memset(w_fb_signal, 0, 1 * sizeof(flag_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_NUMA_DEVICE) {
            dummy_buffer = (bufferElement *) numa_alloc_onnode(BUFFER_SIZE * sizeof(bufferElement), 1);

            r_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 1);
            w_t_signal = (flag_t *) numa_alloc_onnode(1 * sizeof(flag_t), 1);
            w_b_signal = (flag_b *) numa_alloc_onnode(1 * sizeof(flag_b), 1);
            w_d_signal = (flag_d *) numa_alloc_onnode(1 * sizeof(flag_d), 1);
            w_s_signal = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 1);
            w_fb_signal = (flag_s *) numa_alloc_onnode(1 * sizeof(flag_s), 1);

            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_t_signal, 0, 1 * sizeof(flag_t));
            memset(w_b_signal, 0, 1 * sizeof(flag_b));
            memset(w_d_signal, 0, 1 * sizeof(flag_d));
            memset(w_s_signal, 0, 1 * sizeof(flag_s));
            memset(w_fb_signal, 0, 1 * sizeof(flag_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_DRAM) {
            cudaMallocHost(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMallocHost(&r_signal, 1 * sizeof(flag_d));
            cudaMallocHost(&w_t_signal, 1 * sizeof(flag_t));
            cudaMallocHost(&w_b_signal, 1 * sizeof(flag_b));
            cudaMallocHost(&w_d_signal, 1 * sizeof(flag_d));
            cudaMallocHost(&w_s_signal, 1 * sizeof(flag_s));
            cudaMallocHost(&w_fb_signal, 1 * sizeof(flag_s));

            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_t_signal, 0, 1 * sizeof(flag_t));
            memset(w_b_signal, 0, 1 * sizeof(flag_b));
            memset(w_d_signal, 0, 1 * sizeof(flag_d));
            memset(w_s_signal, 0, 1 * sizeof(flag_s));
            memset(w_fb_signal, 0, 1 * sizeof(flag_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        } else if (allocator == CE_UM) {
            cudaMallocManaged(&dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            cudaMallocManaged(&r_signal, 1 * sizeof(flag_d));
            cudaMallocManaged(&w_t_signal, 1 * sizeof(flag_t));
            cudaMallocManaged(&w_b_signal, 1 * sizeof(flag_b));
            cudaMallocManaged(&w_d_signal, 1 * sizeof(flag_d));
            cudaMallocManaged(&w_s_signal, 1 * sizeof(flag_s));
            cudaMallocManaged(&w_fb_signal, 1 * sizeof(flag_s));

            memset(r_signal, 0, 1 * sizeof(flag_d));
            memset(w_t_signal, 0, 1 * sizeof(flag_t));
            memset(w_b_signal, 0, 1 * sizeof(flag_b));
            memset(w_d_signal, 0, 1 * sizeof(flag_d));
            memset(w_s_signal, 0, 1 * sizeof(flag_s));
            memset(w_fb_signal, 0, 1 * sizeof(flag_s));

            memset(dummy_buffer, -1, BUFFER_SIZE * sizeof(bufferElement));
        }

        bufferElement_na *result_g;
        bufferElement_na *result_c;

        cudaMalloc(&result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));
        result_c = (bufferElement_na *) malloc(CPU_NUM_THREADS * sizeof(bufferElement_na));

        if (reader == CE_GPU && writer == CE_GPU) {
            std::cout << "[INFO] Spawning GPU Reader and Writer" << std::endl;

            WriterType *g_spawn_writer;
            WriterType c_spawn_writer = CE_WRITER;

            cudaMalloc(&g_spawn_writer, 1 * sizeof(WriterType));
            cudaMemcpy(g_spawn_writer, &c_spawn_writer, 1 * sizeof(WriterType), cudaMemcpyHostToDevice);

            gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(buffer, dummy_buffer, result_g, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal, g_spawn_writer);

            cudaDeviceSynchronize();

            bufferElement_na * result_h;
            result_h = (bufferElement_na *) malloc(GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na));

            cudaMemcpy(result_h, result_g, GPU_NUM_BLOCKS * GPU_NUM_THREADS * sizeof(bufferElement_na), cudaMemcpyDeviceToHost);

            for (int i = 0; i < GPU_NUM_BLOCKS; i++) {
                for (int j = 0; j < GPU_NUM_THREADS; j++) {
                    if (result_h[i * GPU_NUM_THREADS + j].data > 2000000000) {
                        std::cout << "[" << i << "][" << j << "] " << "--" << std::endl;
                    } else {
                        std::cout << "[" << i << "][" << j << "] " << result_h[i * GPU_NUM_THREADS + j].data << std::endl;
                    }
                }
            }

            free(result_h);

        } else if (reader == CE_CPU && writer == CE_CPU) {
            std::cout << "[INFO] Spawning CPU Reader and Writer" << std::endl;

            WriterType spawn_writer = CE_WRITER;

            std::vector<std::thread> cpu_threads;

            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                cpu_threads.push_back(std::thread(cpu_buffer_reader_writer_propagation_hierarchy, buffer, dummy_buffer, result_c, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal, &spawn_writer));

                CPU_ZERO(&cpuset);
                // if (i == 0) {
                //     CPU_SET(i, &cpuset);
                // } else {
                //     // CPU_SET(i, &cpuset);
                // }
                CPU_SET(i+32, &cpuset);
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
                
                std::cout << result_c[i].data << std::endl;
            }
        } else {
            std::cout << "[INFO] Spawning Heterogeneous Reader and Writer" << std::endl;
            
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
                        
            gpu_buffer_reader_writer_propagation_hierarchy<<<GPU_NUM_BLOCKS, GPU_NUM_THREADS>>>(buffer, dummy_buffer, result_g, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal, g_writer_type);

            std::vector<std::thread> cpu_threads;
            
            for (int i = 0; i < CPU_NUM_THREADS; i++) {
                cpu_threads.push_back(std::thread(cpu_buffer_reader_writer_propagation_hierarchy, buffer, dummy_buffer, result_c, r_signal, w_t_signal, w_b_signal, w_d_signal, w_s_signal, w_fb_signal, c_writer_type));
                
                CPU_ZERO(&cpuset);
                // if (i == 0) {
                //     CPU_SET(i, &cpuset);
                // } else {
                //     // CPU_SET(i, &cpuset);
                // }
                CPU_SET(i+32, &cpuset);
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

            free(result_g_h);
        }

        if (allocator == CE_SYS_MALLOC) {
            free(dummy_buffer);

            free(r_signal);
            free(w_t_signal);
            free(w_b_signal);
            free(w_d_signal);
            free(w_s_signal);
            free(w_fb_signal);
        } else if (allocator == CE_CUDA_MALLOC || allocator == CE_UM) {
            cudaFree(dummy_buffer);

            cudaFree(r_signal);
            cudaFree(w_t_signal);
            cudaFree(w_b_signal);
            cudaFree(w_d_signal);
            cudaFree(w_s_signal);
            cudaFree(w_fb_signal);
        } else if (allocator == CE_NUMA_DEVICE || allocator == CE_NUMA_HOST) {
            numa_free(dummy_buffer, BUFFER_SIZE * sizeof(bufferElement));

            numa_free(r_signal, 1 * sizeof(flag_d));
            numa_free(w_t_signal, 1 * sizeof(flag_t));
            numa_free(w_b_signal, 1 * sizeof(flag_b));
            numa_free(w_d_signal, 1 * sizeof(flag_d));
            numa_free(w_s_signal, 1 * sizeof(flag_s));
            numa_free(w_fb_signal, 1 * sizeof(flag_s));
        } else if (allocator == CE_DRAM) {
            cudaFreeHost(dummy_buffer);

            cudaFreeHost(r_signal);
            cudaFreeHost(w_t_signal);
            cudaFreeHost(w_b_signal);
            cudaFreeHost(w_d_signal);
            cudaFreeHost(w_s_signal);
            cudaFreeHost(w_fb_signal);
        }

        free(result_c);
        cudaFree(result_g);
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
