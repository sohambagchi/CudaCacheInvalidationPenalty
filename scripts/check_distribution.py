# simple program, read a file, each line is a number, make a graph w the numbers on the y1 axis and the line number on the x axis

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy import stats

def analyze_numbers(numbers):
    # print(numbers)
    x = stats.mode(numbers, keepdims=True)[0][0]

    if np.count_nonzero(numbers == x) < len(numbers) * 0.5:
        x = np.median(numbers)
        
    large_numbers = [n for n in numbers if n >= 3 * x]
    
    # print(len(large_numbers))
    if len(large_numbers) > 0:
        large_ratios = [n / x for n in large_numbers]
    else:
        large_ratios = [0]
    
    try:
        avg_ratio = np.mean(large_ratios)
    except:
        avg_ratio = 0
    
    return x, avg_ratio

def main(filename):
    with open(filename, 'r') as f:
        numbers = f.readlines()
        # print(numbers[50])
        
        x = list()
        y1 = list()
        y2 = list()
        
        kernel_starts = list()
        kernel_stops = list()
        
        err_count = 0
        
        buffer_footprint = 0
        data_footprint = 0
        
        for i, line in enumerate(numbers):
            # if '[CPU ITER' in line:
            if 'Size of Buffer' in line:
                buffer_footprint = buffer_footprint + int(line.split(' ')[-5].strip().replace('B', ''))
            if 'Size of Data' in line:
                data_footprint = data_footprint + int(line.split(' ')[-5].strip().replace('B', ''))
            if '[CPU-R]' in line or '[GPU-R]' in line:
                try:
                    y1.append(int(line.split(' ')[6].strip()))
                    x.append(int(line.split(' ')[2].strip()))
                    y2.append(int(line.split(' ')[4].strip()))
                except ValueError:
                    print(line)
                    err_count += 1
            # if '[GPU] -------------------------------' in line:
            if '[GPU-W]' in line or '[CPU-W]' in line:
                if 'Start' in line:
                    kernel_starts.append(len(y1))
                else:
                    kernel_stops.append(len(x))
                    
        # y1 = [(int(_.split(' ')[2].strip())) for _ in numbers if '[CPU ITER' in _]
        # x = [(int(_.split(' ')[1].replace('ITER-', '').replace(']', '').strip())) for _ in numbers if '[CPU ITER' in _]

    # print(numbers)
    
    # print(kernel_starts)
    # print(kernel_stops)

    print(len(kernel_starts))

    # wider graph
    
    plt.figure(figsize=(20, 6))

    plt.plot(x, y1)
    
    avg, avg_slowdown = analyze_numbers(y1)
    
    print(avg)
    # print(large_numbers)
    # print(large_ratios)
    
    # plot y2 on the right side
    
    ax2 = plt.twinx()
    ax2.plot(x, y2, color='orange')
    
    # for i in range(len(kernel_starts)):
        
        # plt.axvspan(xmin = kernel_starts[i], xmax= kernel_stops[i], color='red', alpha=0.5)
    # plt.plot(kernel_starts, [max(y1)]*len(kernel_starts), color='green')
    # plt.plot(kernel_stops, [max(y1)]*len(kernel_stops), color='red')
    
    # put vertical lines at the kernel start and stop
    # plt.axvline(x=kernel_starts, 1,  color='green')
    # plt.axvline(x=kernel_stops, color='red')
    
    for i in range(len(kernel_starts)):
        plt.axvline(x=kernel_starts[i], color='green', linewidth=0.5)
        plt.axvline(x=kernel_stops[i], color='red',    linewidth=0.5)
        
    plt.xticks(range(0, len(x), 500))
        
    plt.xlabel('Iteration')
    
    ax2.set_ylabel('Sum')
    # plt.set_ylabel('Time (ns)')
    
    # plt.ylabel('Time')
    # ax2.set_ylabel('Time (ns)')
    # ax1.set_ylabel('Time (ns)')
    
    plt.title(f"{filename}: reader footprint = {str(buffer_footprint/1024)}KB, data footprint = {str(data_footprint/1024)}KB | avg latency = {avg/8192:.2f} cycles | avg slowdown = {avg_slowdown:.2f}x")
    
    # plt.title(filename + ' - reader footprint: ' + str(buffer_footprint/1024) + 'KB' + ' - data footprint: ' + str(data_footprint/1024) + 'KB | avg latency = ' + str(avg) + " cycles | avg slowdown" + str(avg_slowdown:.2f) + 'cycles')
    
    # explort to png
    plt.show()
    plt.savefig(filename.replace('.txt', '').replace('::', '_') + (str(err_count) if err_count > 0 else '') + '.png')
    
    plt.close()
    
    
if __name__ == '__main__':
    # iterate through all the .txt files in the folder, ignore subfolders
    for file in os.listdir('.'):
        if (file.endswith('.txt') and not file.startswith('result')) and os.path.isfile(file):
            print('Processing ' + file)
            main(file)
            print('Finished ' + file)
            # quit()