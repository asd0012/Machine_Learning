import matplotlib.pyplot as plt
import numpy as np
import random


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

num=30
dataset=rand_seed(2,-3,num)
dataset2=rand_seed(4,-1,num)
dataset3=rand_seed(-5,8,num)
count=0




def check_error(w, dataset):
    result = None
    global count
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) != s:
            result =  x, s
            error += 1
            count += 1
    print  ("error=%s/%s" % (error, len(dataset)))
    return result


def pla(dataset):
    w = np.zeros(3)
    while check_error(w, dataset) is not None:
        x, s = check_error(w, dataset)
        w += s * x
    return w


def draw(dataset,num):
    w = pla(dataset)


    ps = [v[0] for v in dataset]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter([v[1] for v in ps[:int(num/2)]], [v[2] for v in ps[:int(num/2)]],  c='b', marker="o", label='O')
    ax1.scatter([v[1] for v in ps[int(num/2):]], [v[2] for v in ps[int(num/2):]],  c='r', marker="x", label='X')
    l = np.linspace(-2,30)
    a,b = -w[1]/w[2], -w[0]/w[2]
    ax1.plot(l, a*l + b, 'b-')
    plt.legend(loc='upper left');
    plt.show()

if __name__ == '__main__':
    draw(dataset,num)
    
    draw(dataset2,num)

    draw(dataset3,num)    
    
    print("count=",count)
    print("average number of iterations",count/3)