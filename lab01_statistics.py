import numpy as np
from matplotlib import pyplot as plt

data = list()
total = [0]*101
male = [0]*101
female = [0]*101

file = open('/Users/user/Desktop/보윤/충남대/3-1/알고리즘응용/seoul.txt', 'r', encoding='utf-8')
while True:
    line = file.readline()
    if line == '':
        file.close()
        break

    line = line.replace('\n', '') 
    line_tmp = line.replace(" ","").split('\t')
    data.append(line_tmp) #tab을 기준으로 나눠서 리스트 'data'에 저장

    #2번째 index  = 0세, 102번째 index = 100세이상

for l in data:
    if l[1] == "남자":
        for i in range(2, 103):
            male[i-2] += int(l[i])

    elif l[1] == "여자":
        for i in range(2, 103):
            female[i-2] += int(l[i])

for i in range(0, len(male)):
    total[i] = male[i]+female[i] 

x_t = np.arange(len(total))
plt.bar(x_t,total)
plt.show() #total 그래프

x_m = np.arange(len(male))
plt.bar(x_m,total)
plt.show() #male 그래프

x_fe = np.arange(len(female))
plt.bar(x_fe,total)
plt.show() #female 그래프

total_str = "\t".join(map(str, total))
print("계 : " + total_str)
print("계 총합 : " + str(np.sum(total))) #np.sum(total, axis=0)과 동일
print("계 평균 : " + str(int(np.mean(total))))
print("계 분산 : " + str(int(np.var(total))))

mail_str = "\t".join(map(str, male))
print("남자 : " + mail_str)
print("남자 총합 : " + str(np.sum(male)))
print("남자 평균 : " + str(int(np.mean(male))))
print("남자 분산 : " + str(int(np.var(male))))

female_str = "\t".join(map(str, female))
print("여자 : " + female_str)
print("여자 총합 : " + str(np.sum(female))) 
print("여자 평균 : " + str(int(np.mean(female))))
print("여자 분산 : " + str(int(np.var(female))))



