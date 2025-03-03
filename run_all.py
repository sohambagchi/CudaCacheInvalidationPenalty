# simple script to run all .out files in the folder till they complete, and write their results into  .txt file w the same name

import os
import subprocess
import time


scopes = ['cuda::thread_scope_system', 'cuda::thread_scope_device', 'cuda::thread_scope_thread']
sizes = ['uint64_t']#, 'uint32_t', 'uint16_t', 'uint8_t']
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
            
            for reader, writer in (['cpu', 'cpu'], ['cpu', 'gpu'], ['gpu', 'cpu'], ['gpu', 'gpu']):
                if reader == writer == 'gpu':
                    command = ["./" + file, '-m', 'cuda_malloc', '-r', reader, '-w', writer, '-o', f'results{output_file_num}.txt']
                    if file[:-4] + "_cudamalloc_gpu_gpu.txt" in os.listdir():
                        continue
                    print("Running command: " + ' '.join(command))
                    with open(file[:-4] + "_cudamalloc_gpu_gpu.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        output_file_num += 1
                        while p.poll() is None:
                            time.sleep(10)
                        _, err = p.communicate()
                elif reader == writer == 'cpu':
                    command = ["./" + file, '-m', 'malloc', '-r', reader, '-w', writer, '-o', f'results{output_file_num}.txt']
                    if file[:-4] + "_malloc_cpu_cpu.txt" in os.listdir():
                        continue
                    print("Running command: " + ' '.join(command))
                    with open(file[:-4] + "_malloc_cpu_cpu.txt", "w") as f:
                        p = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
                        output_file_num += 1
                        while p.poll() is None:
                            time.sleep(10)
                        _, err = p.communicate()
                else:
                    for mem_alloc in ['dram', 'um']:
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
            
run_all()

