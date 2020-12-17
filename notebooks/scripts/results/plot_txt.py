import os

import matplotlib.pyplot as plt

def parse_line(l):
    print(l.split("]")[-1])
    return float(l.split("]")[-1])

with open("output.txt", "r") as f:
    loss = [parse_line(l) for l in f if not('Epoch' in l) and ']' in l]


def moving_average(l, w=50):
    return [sum(l[i:i+w])/50 for i in range(len(l)-w)]

loss = moving_average(loss)

plt.figure()
plt.plot(loss)
plt.ylabel("Loss")
plt.xlabel("Batch Iteration")
plt.show()