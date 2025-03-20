# simple script to run all .out files in the folder till they complete, and write their results into  .txt file w the same name

import os
import subprocess
import time
import sys


# scopes = ['cuda::thread_scope_system', 'cuda::thread_scope_device', 'cuda::thread_scope_thread']
# sizes = ['uint8_t']#, 'uint32_t', 'uint16_t', 'uint8_t']

# scopes = ['CUDA_THREAD_SCOPE_THREAD', 'CUDA_THREAD_SCOPE_DEVICE', 'CUDA_THREAD_SCOPE_SYSTEM']
# sizes = ['DATA_SIZE_32', 'DATA_SIZE_64'] #'DATA_SIZE_8', 'DATA_SIZE_16']

# prefix = 'cache_invalidation_testing'

# filenames = []

# for scope in scopes:
#     for size in sizes:
#         filenames.append(prefix + '_' + scope + '_' + size + '.out')

allocators = ['malloc', 'dram', 'um', 'numa_host', 'numa_device']

runtime_flags = {
    ('gpu', 'gpu'): ['cuda_malloc'] + allocators,
    ('gpu', 'cpu'): [allocator for allocator in allocators],
    ('cpu', 'gpu'): [allocator for allocator in allocators],
    ('cpu', 'cpu'): [allocator for allocator in allocators]
}

prefix = sys.argv[1]

def run_all():

    for file in os.listdir():
        if file.endswith(".out"):
            print("Running " + file)
            
            for reader, writer in runtime_flags.keys():
                for allocator in runtime_flags[(reader, writer)]:
                    
                    command = ["./" + file, "-m", allocator, "-r", reader, "-w", writer, "-p"]
                    print("Running command: " + ' '.join(command))

                    file_prefix = prefix
                    
                    if 'no_acq' in file:
                        file_prefix += '_no_acq'
                        
                    if 'rel' in file:
                        file_prefix += '_rel'
                        
                    if 'rlx' in file:
                        file_prefix += '_rlx'
                        
                    file_prefix += '_multiproducer'
                        
                    start_time = time.time()
                    with open(file_prefix + f"_{allocator}_{reader}_{writer}.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        while p.poll() is None:
                            time.sleep(10)
                            elapsed_time = time.time() - start_time
                            if elapsed_time > 600:  # 10 minutes
                                p.terminate()
                                print(f"Terminated {command} due to timeout")
                                break
                        _, err = p.communicate()
                        
                    command = ["./" + file, "-m", allocator, "-r", reader, "-w", writer]
                    print("Running command: " + ' '.join(command))

                    file_prefix = prefix
                    
                    if 'no_acq' in file:
                        file_prefix += '_no_acq'
                        
                    if 'rel' in file:
                        file_prefix += '_rel'
                        
                    if 'rlx' in file:
                        file_prefix += '_rlx'
                        
                    file_prefix += '_singleproducer'
                        
                    start_time = time.time()
                    with open(file_prefix + f"_{allocator}_{reader}_{writer}.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        while p.poll() is None:
                            time.sleep(10)
                            elapsed_time = time.time() - start_time
                            if elapsed_time > 60:  # 10 minutes
                                p.terminate()
                                print(f"Terminated {file} due to timeout")
                                break
                        _, err = p.communicate()

            print("Finished " + file)
            
            # break
            
    print("Finished all files")
     

if __name__ == '__main__':
    run_all()
