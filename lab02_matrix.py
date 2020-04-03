import numpy as np
from sklearn.metrics.pairwise import cosine_distances as cd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

def get_manhattan(data1, data2):
    res = 0
    for i in range(0, len(data1)):
        res += abs(float(data1[i]) - float(data2[i]))
    
    return res

def get_euclidean(data1, data2):
    res = 0
    for i in range(0, len(data1)):
        res += (float(data1[i]) - float(data2[i]))**2

    return res**0.5

def show_graph(data): 
    plt.pcolor(data)
    plt.colorbar()
    plt.show()   

def total_distances(data):
    #코사인 거리 
    cos_dist = cd(data, data)
    show_graph(cos_dist)

    #맨하탄 거리
    man_dist = [[0 for col in range(len(data))] for row in range(len(data))]
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            man_dist[i][j] = get_manhattan(data[i], data[j])

    show_graph(man_dist)

    #유클리디안 거리
    eucl_dist = [[0 for col in range(len(data))] for row in range(len(data))]
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            eucl_dist[i][j] = get_euclidean(data[i], data[j])

    show_graph(eucl_dist)


# 파일에서 데이터 읽고 배열 'taxData'에 저장
taxData = []

file = open('/Users/user/Desktop/보윤/충남대/3-1/알고리즘응용/seoul_tax.txt', 'r', encoding='utf-8')
lines = file.readlines()
del lines[0]
for line in lines:
    line = line.replace('\n', '')
    line = line.replace(" ", "")
    line = line.split('\t')
    del line[0]
    taxData.append(line)

#distances
total_distances(taxData)

#정규화
scaler = MinMaxScaler()
taxData_norm = scaler.fit_transform(taxData)

total_distances(taxData_norm)
