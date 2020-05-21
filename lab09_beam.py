import numpy as np

def greedy_search(col, row):
    score = 10.0
    seq = []

    for i in range(col):
        np.random.seed(i)
        data = np.random.rand(row)
        # print(data)
        m = max(data)
        mi = -1
        for j in range(row):
            if data[j]==m:
                mi = j
                break
        score *= m
        seq.append(mi)

    return [seq, score]
        

def beam_search(col, row, k):
    score = 10.0
    seq = []
    res = [[seq, score]]

    for d in range(col):
        np.random.seed(d)
        data = np.random.rand(row)
        all_candidates = list()

        for i in range(len(res)):
            se, sc = res[i]
            for j in range(row):
                candidate = [se + [j], sc * data[j]] #[순서, 1.0*-log(data의 j번째 요소)]
                all_candidates.append(candidate) #계산된 후보를 list에 삽입
        
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        res = ordered[:k]

    return res



if __name__=='__main__':
    print("Greedy Search")
    print(greedy_search(10, 5))
    print("Beam Search")
    beam = beam_search(10,5,3)
    for b in beam:
        print(b)


