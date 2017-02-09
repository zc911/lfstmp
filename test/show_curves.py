import matplotlib.pyplot as plt
import sys

def read_data(filename):
    name = None
    values = []
    with open(filename, 'r') as f:
        step =  1 #10
        count = 0
        for line in f:
            if name is None:
                name = line.strip()
            else:
                count = count + 1
                if not (count % step): 
                    values.append(line.strip().split('\t'))
    return name, zip(*values)

#FILE_LIST = ['roc_points.log']

def main(FILE_LIST):
    fig, ax = plt.subplots()
    for path in FILE_LIST:
        name, (x, y) = read_data(path)
        #print len(x), len(y)
        ax.plot(x, y, '-', linewidth=2, label=name)
        #ax.plot(x, y, '-o', linewidth=2, label=name)
    ax.legend(loc='lower right')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    plt.savefig('test.png', dpi=360)

if __name__ == "__main__":
    FILE_LIST = sys.argv[1:]
    main(FILE_LIST)
