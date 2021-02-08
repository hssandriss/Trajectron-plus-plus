import json
import os
import numpy as np
import matplotlib.pyplot as plt

file_name = "accuracies_g_0.001_64_relu_random_28.json"
values_tag = "accuracy per class"
val_type = values_tag.split(" ")[0]
figdir = "plots/"
figdir_name = file_name[:-5]
allfigs_dir = os.path.join(figdir, figdir_name)
os.makedirs(allfigs_dir, exist_ok=True)

jsonfile = json.loads(open(file_name).read().encode('utf-8'))

nb_classes = len(jsonfile[0][values_tag])

avg_val = lambda x: np.mean([x[values_tag][str(i)] for i in range(nb_classes)])
class_val = lambda x,i: x[values_tag][str(i)]

values = [avg_val(x) for x in jsonfile]
# plt.figure()
plt.plot(values)
plt.savefig(os.path.join(allfigs_dir, f'{val_type}_mean.png'))
plt.clf()

for c in range(nb_classes):
    values= [class_val(x, c) for x in jsonfile]
    plt.plot(values)
    plt.savefig(os.path.join(allfigs_dir,f'{val_type}_{c}.png'))
    plt.clf()
