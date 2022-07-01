import matplotlib.pyplot as plt
import numpy as np
import operator
import random
import pla
from pla import draw

def rand_seed(m, b, num=2):
    # create empty list
    x_coor = []
    y_coor = []
    pla_data=[]
    # positive and negtive point number
    pos_num = int(num / 2)
    neg_num = num - pos_num
    # random create point
    for i in range(pos_num):
        x = random.randint(0, 30)
        r = random.randint(1, 30)
        y = m * x + b - r     
        x_coor.append(x)
        y_coor.append(y)   
        label=(1 if m >= 0 else -1)
        pla_data.append([(1,x,y),label])

    for i in range(neg_num):
        x = random.randint(0, 30)
        r = random.randint(1, 30)
        y = m * x + b + r
        x_coor.append(x)
        y_coor.append(y)
        label=(-1 if m >= 0 else 1)
        pla_data.append([(1,x,y),label])
    return pla_data

num=2000
dataset=rand_seed(2,9,num=2000)

count=0 
 
def dot(*v):
    return sum(map(operator.mul, *v))
 
 
def sign(v):
    if v > 0:
        return 1
    elif v == 0:
        return 0
    else:  # v < 0
        return -1
 
 
def check_error(w, x, y):
    if sign(dot(w, x)) != y:
        return True
    else:
        return False
 
 
def update(w, x, y):
    u = map(operator.mul, [y] * len(x), x)
    w = map(operator.add, w, u)
    return list(w)
 
 
def sum_errors(w, dataset):
    errors = 0
    for x, y in dataset:
        if check_error(w, x, y):
            errors += 1
 
    return errors
 
 
def pocket(dataset):
    # 初始化 w
    w = [0] * 3
    min_e = sum_errors(w, dataset)
 
    max_t = 300
    for t in range(0, max_t):
        wt = None
        et = None
 
        while True:
            x, y = random.choice(dataset)
            if check_error(w, x, y):
                wt = update(w, x, y)
                et = sum_errors(wt, dataset)
                break
 
        if et < min_e:
            w = wt
            min_e = et
 
        print("{}: {}".format(t, tuple(w)))
        print("min erros: {}".format(min_e))
 
        t += 1
 
        if min_e == 0:
            break
 
    return (w, min_e)
 
 

def main():
    w, e = pocket(list(dataset))
 
    fig = plt.figure()
 
    ax1 = fig.add_subplot(111)
 
    xx = list(filter(lambda d: d[1] == -1, dataset))
    ax1.scatter([x[0][1] for x in xx], [x[0][2] for x in xx],
                s=100, c='b', marker="x", label='-1')
    oo = list(filter(lambda d: d[1] == 1, dataset))
    ax1.scatter([x[0][1] for x in oo], [x[0][2] for x in oo],
                s=100, c='r', marker="o", label='1')
    l = np.linspace(-2, 50)
 

    if w[2]:
        a, b = -w[1] / w[2], -w[0] / w[2]
        ax1.plot(l, a * l + b, 'b-')
    else:
        ax1.plot([-w[0] / w[1]] * len(l), l, 'b-')
 
    plt.legend(loc='upper left', scatterpoints=1)
    plt.show()
 
 
if __name__ == '__main__':
    
    main()
        
    draw(dataset,num)

