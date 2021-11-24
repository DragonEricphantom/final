# -*- coding: utf-8 -*-
"""
动态规划法
name:JCH
date:6.8
"""

import pandas as pd
import numpy as np
import math
import time
import darknet_image_mod as dim
from itertools import permutations

"""
N:城市数
s:二进制表示，遍历过得城市对应位为1，未遍历为0
dp:动态规划的距离数组
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
path:记录下一个应该到达的城市
"""

def FW(dist, middlenode):
    for k in range(len(dist)):
        for i in range(len(dist)):
            for j in range(len(dist)):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    middlenode[i][j] = k

def TSP(dist, visit, N):
    p = np.ones((math.factorial(N-1), 1))
    per = np.ones((math.factorial(N - 1), N))
    for i in range(math.factorial(N-1)):
        p[i] = np.array([visit[0]])

    result = permutations(visit[1:])
    index = 0
    for answer in result:
        per[index] = np.concatenate((np.array(p)[index], np.array(answer)))
        index += 1

    sumpath = 100000000000
    for i in per:
        d = 0
        for j in range(len(i) - 1):
            # print('i[j] =', i[j])
            # print('visit.index(i[j]) =' , visit.index(i[j]))
            # print('visit.index(i[j+1]) =', visit.index(i[j+1]))
            d += dist[visit.index(i[j])][visit.index(i[j+1])]
            # d += dist[visit[int(np.where(i == i[j])[0][0])] - 1][visit[int(np.where(i == i[j+1])[0][0])] - 1]
        # print('path:', i, ' distance:', d)
        if d < sumpath:
            sumpath = d
            minroute = i
    return minroute, sumpath

def printpath(middlenode, node1, node2, depth):
    if depth == 0:
        #print('depth = ', depth)
        print(int(node1 + 1))

    if middlenode[int(node1)][int(node2)] != -1:
        depth += 1
        printpath(node1, middlenode[int(node1)][int(node2)], depth)
        printpath(middlenode[int(node1)][int(node2)], node2, depth)
        if depth == 0:
            print(int(node2 + 1))
        depth -= 1
    else:
        print(int(node2 + 1))

def path(visit):
    dataframe = pd.read_csv("TSP8cities_with_weight_assigned.tsp", sep=" ", header=None)
    num_of_all_spot = len(dataframe) - 1
    v = dataframe.iloc[1:num_of_all_spot + 1, 1:num_of_all_spot + 1]

    print('dataframe: \n', dataframe)
    print('num_of_all_spot: \n', num_of_all_spot)
    print('v: \n', v)

    train_v = np.array(v)
    # train_d = train_v
    all_point = np.zeros((num_of_all_spot, num_of_all_spot))
    time_point = np.zeros((num_of_all_spot, num_of_all_spot))
    middlenode = np.zeros((num_of_all_spot, num_of_all_spot), dtype=np.int)
    depth = 0  # recursion depth

    print(visit)

    population = []
    dim.YOLO(population)

    num_of_spot = len(visit)
    dist = np.zeros((num_of_spot, num_of_spot))

    path = np.ones((2 ** (num_of_spot + 1), num_of_spot))
    dp = np.ones((2 ** (num_of_spot + 1), num_of_spot)) * -1

    # Initialize
    for i in range(num_of_all_spot):
        for j in range(num_of_all_spot):
            all_point[i][j] = train_v[i][j]
            middlenode[i][j] = -1

    print('All point: ')
    print(all_point)


    # Add population
    for i in range(num_of_spot):
        for j in range(num_of_spot):
            if population[i] > 30:
                all_point[j][i] += 15
            elif population[i] > 20:
                all_point[j][i] += 10
            else:
                all_point[j][i] += 5


    FW(all_point, middlenode)

    time_point = all_point/2

    print('time', time_point)

    # pick up spot want to visit
    for i in range(num_of_spot):
        for j in range(num_of_spot):
            dist[i][j] = all_point[visit[i] - 1][visit[j] - 1]

    print('dist: ')
    print(dist)

    minpath, mindistance = TSP(dist, visit, len(visit))
    print('path = ', minpath)
    print('distance = ', mindistance)

    total_time = 0
    for i in range(len(minpath) - 1):
        total_time = total_time + time_point[int(i)][int(i+1)]
    print('time = ', total_time)
    return minpath

    """
    current_spot = minpath[0]
    next_spot = minpath[1]

    visit.remove(current_spot)
    move = 0
    for i in visit:
        if i == next_spot:
            break
        else:
            move += 1
    for i in range(move):
        visit.insert(len(visit), visit[0])
        visit.remove(visit[0])
    """

'''
if __name__ == "__main__":
    dataframe = pd.read_csv("TSP8cities_with_weight_assigned.tsp", sep=" ", header=None)
    num_of_all_spot = len(dataframe) - 1
    v = dataframe.iloc[1:num_of_all_spot + 1, 1:num_of_all_spot + 1]

    train_v = np.array(v)
    train_d = train_v
    all_point = np.zeros((num_of_all_spot, num_of_all_spot))
    middlenode = np.zeros((num_of_all_spot, num_of_all_spot), dtype=np.int)
    depth = 0  # recursion depth

    visit = []
    for i in range(num_of_all_spot):
        visit.append(i+1)
    times = 1

    print(visit)

    while len(visit) > 1:
        population = []
        dim.YOLO(population)

        print('-----------------------', times, '-----------------------')
        times += 1
        print('visit = ', visit)
        num_of_spot = len(visit)
        dist = np.zeros((num_of_spot, num_of_spot))

        path = np.ones((2 ** (num_of_spot + 1), num_of_spot))
        dp = np.ones((2 ** (num_of_spot + 1), num_of_spot)) * -1

        # Initialize
        for i in range(num_of_all_spot):
            for j in range(num_of_all_spot):
                all_point[i][j] = train_v[i][j]
                middlenode[i][j] = -1

        print('All point: ')
        print(all_point)

        """
        # Add population
        for i in range(num_of_spot):
            for j in range(num_of_spot):
                if population[i] > 30:
                    all_point[j][i] += 15
                elif population[i] > 20:
                    all_point[j][i] += 10
                else:
                    all_point[j][i] += 5
        """

        FW(all_point, middlenode)

        # pick up spot want to visit
        for i in range(num_of_spot):
            for j in range(num_of_spot):
                dist[i][j] = all_point[visit[i] - 1][visit[j] - 1]

        print('dist: ')
        print(dist)

        minpath, mindistance = TSP(visit, len(visit))
        print('path = ', minpath)
        print('distance = ', mindistance)

        current_spot = minpath[0]
        next_spot = minpath[1]

        visit.remove(current_spot)
        move = 0
        for i in visit:
            if i == next_spot:
                break
            else:
                move += 1
        for i in range(move):
            visit.insert(len(visit), visit[0])
            visit.remove(visit[0])
'''



"""
结果：
distance:  35.0
path:
1
3.0
4.0
2.0
程序的运行时间是：0.004149800000959658
"""