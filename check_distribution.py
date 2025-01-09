# simple program, read a file, each line is a number, make a graph w the numbers on the y1 axis and the line number on the x axis

import matplotlib.pyplot as plt
import sys
import os

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
        
        for i, line in enumerate(numbers):
            if '[CPU ITER' in line:
                try:
                    y1.append(int(line.split(' ')[2].strip()))
                    x.append(int(line.split(' ')[1].replace('ITER-', '').replace(']', '').strip()))
                    y2.append(int(line.split(' ')[3].strip()))
                except ValueError:
                    print(line)
                    err_count += 1
            if '[GPU] -------------------------------' in line:
                if 'Triggering' in line:
                    kernel_starts.append(len(y1))
                else:
                    kernel_stops.append(len(x))
                    
        # y1 = [(int(_.split(' ')[2].strip())) for _ in numbers if '[CPU ITER' in _]
        # x = [(int(_.split(' ')[1].replace('ITER-', '').replace(']', '').strip())) for _ in numbers if '[CPU ITER' in _]

    # print(numbers)
    
    print(kernel_starts)
    print(kernel_stops)

    # wider graph
    
    plt.figure(figsize=(20, 6))

    plt.plot(x, y1)
    
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
    
    plt.ylabel('Time (ns)')
    
    plt.title(filename)
    
    # explort to png
    plt.show()
    plt.savefig(filename.replace('.txt', '').replace('::', '_') + (str(err_count) if err_count > 0 else '') + '.png')
    
    plt.close()
    
    
if __name__ == '__main__':
    # iterate through all the .txt files in the folder, ignore subfolders
    for file in os.listdir('.'):
        if file.endswith('.txt') and os.path.isfile(file):
            print('Processing ' + file)
            main(file)
            print('Finished ' + file)