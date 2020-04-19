from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

def read_file(path):
    data = []
    file = open(path, 'r', encoding='utf-8')
    lines = file.readlines()
    del lines[0]
    for line in lines:
        line = line.replace('\n', '')
        line = line.split('\t')
        data.append(line)
    
    return data 

def draw_graph(data, kind):
    if kind==1:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], cmap='rainbow')
        plt.show()
    
    elif kind==2:
        plt.figure()
        plt.scatter(data, [0]*len(data), cmap='rainbow')
        plt.show()

def sklearn_pca(data):
    X = np.array(data)
    pca = PCA(n_components=1)
    res = pca.fit_transform(X)
    draw_graph(res, 2)

def pca(data):
    #covariance matrix 구하기
    data_t = np.transpose(data)
    cov = []
    for i in range(0,2):
        cov_tmp = []
        for j in range(0,2):
            c = covariance_matrix(data_t[i], data_t[j])
            cov_tmp.append(c)
        cov.append(cov_tmp)
    cov = np.array(cov)

    #고유값, 교유벡터 구하기
    e_val, e_vec = LA.eig(cov)

    #고유값의 크기순으로 내림차순 정렬
    if np.argsort(e_val)[1] == 1 :
        e_val[[1, 0]] = e_val[[0, 1]]
        e_vec.T[[1, 0]] = e_vec.T[[0, 1]]
        
    #projection
    res = np.dot(data, e_vec.T[0])
    draw_graph(res, 2)

def covariance_matrix(data1, data2):
    d1 = data1 - mean(data1)
    d2 = data2 - mean(data2)
    cov = np.dot(d1, np.transpose(d2)) / len(data1)

    return cov

def mean(data):
    sum = 0
    for d in data:
        sum += d
    return sum / len(data)

if __name__ == '__main__':
    studentData = read_file('./seoul_student.txt')

    #정규화
    scaler = MinMaxScaler()
    studentData_norm = scaler.fit_transform(studentData)
    draw_graph(studentData_norm, 1)

    #sklearn pca
    sklearn_pca(studentData_norm)

    #구현한 pca
    pca(studentData_norm)

