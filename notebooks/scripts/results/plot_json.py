import os
import json
import numpy as np
import matplotlib.pyplot as plt

with open("history.json", "r") as fd:
    history = json.load(fd)

def moving_average(l, w=50):
    return [sum(l[i:i+w])/50 for i in range(len(l)-w)]

loss = np.concatenate(history["loss_train"], axis=0)

print(np.shape(loss))

loss = moving_average(loss)

plt.figure()
plt.plot(loss)
plt.ylabel("Loss")
plt.xlabel("Batch Iteration")
plt.show()