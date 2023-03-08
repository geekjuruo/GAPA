import os
sum = 0
count = 0
path = './results'
for file in os.listdir(path):
    count += 1
    with open(os.path.join(path, file), encoding='utf-8') as f:
        a = f.readlines()
        sum += float(a[2])
print(sum/count)
print(count)
