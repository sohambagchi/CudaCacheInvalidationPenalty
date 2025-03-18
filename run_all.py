# simple script to run all .out files in the folder till they complete, and write their results into  .txt file w the same name

import os
import subprocess
import time
import sys


# scopes = ['cuda::thread_scope_system', 'cuda::thread_scope_device', 'cuda::thread_scope_thread']
# sizes = ['uint8_t']#, 'uint32_t', 'uint16_t', 'uint8_t']

scopes = ['CUDA_THREAD_SCOPE_THREAD', 'CUDA_THREAD_SCOPE_DEVICE', 'CUDA_THREAD_SCOPE_SYSTEM']
sizes = ['DATA_SIZE_32', 'DATA_SIZE_64'] #'DATA_SIZE_8', 'DATA_SIZE_16']

prefix = 'cache_invalidation_testing'

filenames = []

for scope in scopes:
    for size in sizes:
        filenames.append(prefix + '_' + scope + '_' + size + '.out')


def run_all():
    
    output_file_num = 0
    
    for file in filenames:
        if file.endswith(".out"):
            print("Running " + file)
            #(['cpu', 'cpu'], ['cpu', 'gpu'], ['gpu', 'cpu'], ['gpu', 'gpu']):
            # for reader, writer in (['gpu', 'gpu']): 
            reader = 'gpu'
            writer = 'gpu'
            if reader == writer == 'gpu':
                mem_alloc = ['cuda_malloc', 'malloc']#, 'dram', 'numa_device', 'numa_host']
                for mem in mem_alloc:
                    if f"gpu_gpu_{file.split('_')[9].replace('.out', '')}_{file.split('_')[6]}" in os.listdir():
                        continue
                    command = ["./" + file, '-m', mem, '-r', reader, '-w', writer]
                    # if file[:-4] + f"_{mem}_gpu_gpu.txt" in os.listdir():
                    #     continue
                    print("Running command: " + ' '.join(command))
                    with open(file[:-4] + f"_{mem}_gpu_gpu.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        output_file_num += 1
                        while p.poll() is None:
                            time.sleep(10)
                        _, err = p.communicate()
            elif reader == writer == 'cpu':
                mem_alloc = ['malloc', 'dram', 'numa_host', 'numa_device']
                for mem in mem_alloc:
                    command = ["./" + file, '-m', mem, '-r', reader, '-w', writer, '-o', f'results{output_file_num}.txt']
                    if file[:-4] + f"_{mem}_cpu_cpu.txt" in os.listdir():
                        continue
                    print("Running command: " + ' '.join(command))
                    with open(file[:-4] + f"_{mem}_cpu_cpu.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        output_file_num += 1
                        while p.poll() is None:
                            time.sleep(10)
                        _, err = p.communicate()
            else:
                for mem_alloc in ['dram', 'um', 'malloc', 'numa_host', 'numa_device']:
                    command = ["./" + file, '-m', mem_alloc, '-r', reader, '-w', writer, '-o', f'results{output_file_num}.txt']
                    if file[:-4] + f"_{mem_alloc}_{reader}_{writer}.txt" in os.listdir():
                        continue
                    print("Running command: " + ' '.join(command))
                    with open(file[:-4] + f"_{mem_alloc}_{reader}_{writer}.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        output_file_num += 1
                        while p.poll() is None:
                            time.sleep(10)
                        _, err = p.communicate()
            
            # with open(file[:-4] + "_malloc.txt", "w") as f:
            #     p = subprocess.Popen(["./" + file, '-m', 'malloc'], stdout=f, stderr=subprocess.PIPE)
            #     while p.poll() is None:
            #         time.sleep(1)
            #     _, err = p.communicate()
            
            # with open(file[:-4] + "_numa_host.txt", "w") as f:
            #     p = subprocess.Popen(["./" + file, '-m', 'numa_host'], stdout=f, stderr=subprocess.PIPE)
            #     while p.poll() is None:
            #         time.sleep(1)
            #     _, err = p.communicate()
                
            # with open(file[:-4] + "_numa_device.txt", "w") as f:
            #     p = subprocess.Popen(["./" + file, '-m', 'numa_device'], stdout=f, stderr=subprocess.PIPE)
            #     while p.poll() is None:
            #         time.sleep(1)
            #     _, err = p.communicate()
                
            # with open(file[:-4] + "_dram.txt", "w") as f:
            #     p = subprocess.Popen(["./" + file, '-m', 'dram'], stdout=f, stderr=subprocess.PIPE)
            #     while p.poll() is None:
            #         time.sleep(1)
            #     _, err = p.communicate()
                
            # with open(file[:-4] + "_um.txt", "w") as f:
            #     p = subprocess.Popen(["./" + file, '-m', 'um'], stdout=f, stderr=subprocess.PIPE)
            #     while p.poll() is None:
            #         time.sleep(1)
            #     _, err = p.communicate()
                
            print("Finished " + file)
     
     
            
if len(sys.argv) > 1 and sys.argv[1] == 'test':
    print("testing")
    for file in filenames:
        if 'system' in file:
            for reader, writer in [['gpu', 'gpu'], ['cpu', 'cpu'], ['cpu', 'gpu'], ['gpu', 'cpu']]:
                command = ["./" + file, '-m', 'malloc', '-r', reader, '-w', writer, '-o', f'results.txt']
                        # if file[:-4] + f"{reader}_{writer}.txt" in os.listdir():
                        #     continue
                print("Running command: " + ' '.join(command))
                with open(f"{reader}_{writer}.txt", "w") as f:
                    p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                    while p.poll() is None:
                        time.sleep(10)
                    _, err = p.communicate()
elif len(sys.argv) > 1 and sys.argv[1] == 'profile':
    print("profiling")
    for file in filenames:
        reader = 'gpu'
        writer = 'gpu'
        command = ["ncu", "-o", f"./gpu_gpu_{file.split('_')[9].replace('.out', '')}_{file.split('_')[6]}", "--set", "full", "./" + file, '-m', 'malloc', '-r', reader, '-w', writer]
                # if file[:-4] + f"{reader}_{writer}.txt" in os.listdir():
                #     continue
        print("Running command: " + ' '.join(command))
        with open(f"{reader}_{writer}_{file.split('_')[9].replace('.out', '')}_{file.split('_')[6]}_profile.txt", "w") as f:
            p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
            while p.poll() is None:
                time.sleep(10)
            _, err = p.communicate()
else:
    run_all()

