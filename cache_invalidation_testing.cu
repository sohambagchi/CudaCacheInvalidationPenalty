#include "cache_events.cuh"

int main(int argc, char* argv[]) {

    if (argc < 8) {
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

    std::string filename;

    int opt;

    while ((opt = getopt(argc, argv, "m:r:w:o:")) != -1) {
        switch (opt) {
            case 'o':
                filename = optarg;
                break;
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
            default:
                std::cout << "Error: Invalid argument" << std::endl;
                return 0;
           }
        }
    }

    const AllocatorType allocator = allocator_t;
    const ReaderWriterType reader = reader_t;
    const ReaderWriterType writer = writer_t;

    cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *buffer;

    if (allocator == CE_SYS_MALLOC) {
        std::cout << "[INFO] Allocating buffer using system malloc" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) malloc(BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>));
    } else if (allocator == CE_NUMA_HOST) {
        std::cout << "[INFO] Allocating buffer using numa_alloc_onnode (cpu)" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) numa_alloc_onnode(BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>), 0);
    } else if (allocator == CE_NUMA_DEVICE) {
        std::cout << "[INFO] Allocating buffer using numa_alloc_onnode (gpu)" << std::endl;
        buffer = (cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE> *) numa_alloc_onnode(BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>), 1);
    } else if (allocator == CE_DRAM) {
        std::cout << "[INFO] Allocating buffer using cudaMallocHost" << std::endl;
        cudaMallocHost(&buffer, BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>));
    } else if (allocator == CE_UM) {
        std::cout << "[INFO] Allocating buffer using cudaMallocManaged" << std::endl;
        cudaMallocManaged(&buffer, BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>));
    } else if (allocator == CE_CUDA_MALLOC) {
        std::cout << "[INFO] Allocating buffer using cudaMalloc" << std::endl;
        cudaMalloc(&buffer, BUFFER_SIZE * sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>));
    }

    std::cout << "[INFO] Size of Buffer: " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE << "B | " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE / 1024 << "KB | " << sizeof(cuda::atomic<DATA_SIZE, CUDA_THREAD_SCOPE>) * BUFFER_SIZE / 1024 / 1024 << "MB" << std::endl;

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

    std::cout << "[TIME] Baseline Writer Duration" << std::endl;
    
    const int chunkSize = BUFFER_SIZE / 16;
    const int threadsPerBlock = 1;
    const int totalThreads = (BUFFER_SIZE + chunkSize - 1) / chunkSize;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    if (writer == CE_GPU) {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        // gpu_buffer_writer<<<blocksPerGrid, threadsPerBlock>>>(buffer, 1, chunkSize);
        gpu_buffer_writer_single_iter<<<blocksPerGrid, threadsPerBlock>>>(buffer, chunkSize);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "[TIME-GPU] " << milliseconds << "ms" << std::endl;
    } else if (writer == CE_CPU) {
        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i].store(1, cuda::memory_order_relaxed);
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        std::cout << "[TIME-CPU] " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
    }   

    std::cout << "[TIME] Baseline Reader Duration" << std::endl;
    
    /****
        * 
        * If reader is GPU, and writer is GPU, store the clock cycles
        * If reader is CPU, and writer is CPU, store the nanoseconds
        * If reader is GPU, and writer is CPU, convert the nanoseconds to clock cycles
        * If reader is CPU, and writer is GPU, convert the clock cycles to nanoseconds
        * 
        * OR just have both values no matter what.
        ****/

    clock_t gpu_reader_cycles;
    uint64_t cpu_reader_ns;


    if (reader == CE_GPU) {

        DATA_SIZE * result;

        cudaMalloc(&result, 1 * sizeof(unsigned int));

        clock_t *gpu_reader_duration;
        cudaMalloc(&gpu_reader_duration, 1 * sizeof(clock_t));
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);

        // cudaEventRecord(start);
        gpu_buffer_reader_single_iter<<<1, 1>>>(buffer, result, gpu_reader_duration);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);

        cudaDeviceSynchronize();

        clock_t *h_gpu_reader_duration;
        h_gpu_reader_duration = (clock_t *) malloc(1 * sizeof(clock_t));
        cudaMemcpy(h_gpu_reader_duration, gpu_reader_duration, 1 * sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        gpu_reader_cycles = *h_gpu_reader_duration;
        cpu_reader_ns = *h_gpu_reader_duration * 1000000 / get_gpu_properties();

        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // reader_time = milliseconds * 1000000; // convert to nanoseconds

        std::cout << "[TIME-GPU] " << cpu_reader_ns << "ns | " << gpu_reader_cycles << " cycles" << std::endl;
    } else if (reader == CE_CPU) {
        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i].load();
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        cpu_reader_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        gpu_reader_cycles = cpu_reader_ns * get_gpu_properties() / 1000000;

        std::cout << "[TIME-CPU] " << cpu_reader_ns << "ns | " << gpu_reader_cycles << " cycles" << std::endl;
    }

    // we want the writer to wait for 10 reader iterations
    if (writer == CE_GPU) {
        gpu_reader_cycles *= 100;
        cpu_reader_ns *= 100;
    } else {
        gpu_reader_cycles *= 10;
        cpu_reader_ns *= 10;
    }
    // gpu_reader_cycles *= 10;
    // cpu_reader_ns *= 10;
    
    struct timespec *ts;
    ts = (struct timespec *) malloc(sizeof(struct timespec));
    
    ts->tv_sec = cpu_reader_ns / 1000000000;
    ts->tv_nsec = cpu_reader_ns % 1000000000;

    std::cout << std::endl;
    
    cpu_set_t cpuset;
    std::thread reader_thread;
    std::thread writer_thread;

    cudaStream_t stream_r, stream_w;
    cudaStreamCreate(&stream_r);
    cudaStreamCreate(&stream_w);

    void * durations;

    DATA_SIZE * result;

    std::cout << "[INFO] Spawning Reader" << std::endl;

    if (reader == CE_GPU) {
        cudaMalloc(&result, NUM_ITERATIONS * sizeof(DATA_SIZE));
        
        cudaMalloc(&durations, NUM_ITERATIONS * sizeof(clock_t));

        gpu_buffer_reader<<<1,1,0,stream_r>>>(buffer, result, (clock_t *) durations);
    } else if (reader == CE_CPU) {
        result = (DATA_SIZE *) malloc(NUM_ITERATIONS * sizeof(DATA_SIZE));

        durations = (std::chrono::duration<uint64_t, std::nano> *) malloc(NUM_ITERATIONS * sizeof(std::chrono::duration<uint64_t, std::nano>)); 
        reader_thread = std::thread(cpu_buffer_reader, buffer, result, (std::chrono::duration<uint64_t, std::nano> *) durations);

        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(reader_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    sleep(1);

    std::cout << "[INFO] Spawning Writer" << std::endl;
    
    clock_t *reader_time_g;
    cudaMalloc(&reader_time_g, 1 * sizeof(clock_t));
    
    if (writer == CE_GPU) {

        cudaMemcpy(reader_time_g, &gpu_reader_cycles, 1 * sizeof(clock_t), cudaMemcpyHostToDevice);
        
        if (reader == CE_GPU) {
            gpu_buffer_writer<<<blocksPerGrid, threadsPerBlock, 0, stream_w>>>(buffer, chunkSize, reader_time_g);
        } else if (reader == CE_CPU) {
            gpu_buffer_writer_single_thread<<<1,1, 0, stream_w>>>(buffer, chunkSize, reader_time_g);
        }
        
    } else if (writer == CE_CPU) {
        
        writer_thread = std::thread(cpu_buffer_writer, buffer, ts);
        
        CPU_ZERO(&cpuset);
        CPU_SET(1, &cpuset);
        pthread_setaffinity_np(writer_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    std::cout << "[INFO] Waiting for threads to finish" << std::endl;
    
    if (reader == CE_CPU) {
        reader_thread.join();
    } 
    
    if (writer == CE_CPU) {
        writer_thread.join();
    }
    
    if (reader == CE_GPU || writer == CE_GPU) {
        cudaDeviceSynchronize();
    }
    
    std::cout << "[INFO] Preparing results" << std::endl;
    
    DATA_SIZE * result_host = (DATA_SIZE *) malloc(NUM_ITERATIONS * sizeof(DATA_SIZE));
    clock_t * durations_host = (clock_t *) malloc(NUM_ITERATIONS * sizeof(clock_t));
    
    if (reader == CE_GPU) {
        cudaMemcpy(result_host, result, NUM_ITERATIONS * sizeof(DATA_SIZE), cudaMemcpyDeviceToHost);
        cudaMemcpy(durations_host, durations, NUM_ITERATIONS * sizeof(clock_t), cudaMemcpyDeviceToHost);
    } else if (reader == CE_CPU) {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result_host[i] = result[i];
        }
        // std::cout << "Copying durations" << std::endl;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            durations_host[i] = ((std::chrono::duration<uint64_t, std::nano> *) durations)[i].count();
        }
    }
    
    std::cout << "[INFO] Results" << std::endl;
    
    FILE * file = fopen(filename.c_str(), "w");
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // std::cout << "[" << i << "] " << result_host[i] << " | " << durations_host[i] << std::endl;
        fprintf(file, "[%d] %ld %ld\n", i, result_host[i], durations_host[i]);
    }
    
    fclose(file);
    
    cudaStreamDestroy(stream_r);
    cudaStreamDestroy(stream_w);
    
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
    
    if (reader == CE_GPU) {
        cudaFree(result);
        cudaFree(durations);
    } else if (reader == CE_CPU) {
        free(result);
        free(durations);
    }
    
    cudaFree(reader_time_g);
    free(result_host);
    free(durations_host);

    return 0;
}
