"""File used for extracting 50 best feature combinations"""
import csv

results = []
with open('results_lgm.csv','r') as f:
    re = csv.reader(f,delimiter=',')
    for row in re:
        a = []
        for i,j in enumerate(row):
            if i==0:
                a.append(float(j))
            else:
                a.append(j)
        results.append(a)

# Sort the results by accuracy in descending order
results.sort(key=lambda x: x[0], reverse=True)

results = results[:50]
with open('features_lgm.csv', 'a', newline='') as fil:
    writer = csv.writer(fil)
    for i in results:
        x = i[1]
        y = []
        for i in x:
            if i>= '0' and i<='9':
                y.append(int(i))
        writer.writerow(y)

