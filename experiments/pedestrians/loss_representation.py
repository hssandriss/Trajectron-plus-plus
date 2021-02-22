import json
import matplotlib.pyplot as plt

file_loss = "/misc/lmbraid21/ayadim/Trajectron-plus-plus/experiments/pedestrians/models/models_Multi_hyp16_Jan_2021_19_56_41_hotel_ar3/losses_hotel_ar3_binary_lr01_C04_coswarm.json"
file_acc = "/misc/lmbraid21/ayadim/Trajectron-plus-plus/experiments/pedestrians/models/models_Multi_hyp16_Jan_2021_19_56_41_hotel_ar3/accuracies_hotel_ar3_binary_lr01_C04_coswarm.json"

loss_json = json.loads(open(file_loss).read().encode('utf-8'))
acc_json = json.loads(open(file_acc).read().encode('utf-8'))

nb_classes = len(loss_json[0])

loss_maj = []
loss_min = []
acc_maj = []
acc_min = []
for i in range(len(loss_json)):
    loss_maj.append(loss_json[i]['0'])
    loss_min.append(loss_json[i][list(loss_json[i].keys())[-1]])
    acc_maj.append(acc_json[i]['0'])
    acc_min.append(acc_json[i][list(loss_json[i].keys())[-1]])
    
import pdb; pdb.set_trace()

plt.plot(loss_maj)
plt.savefig('loss_majority.png')
plt.clf()
plt.plot(loss_min)
plt.savefig('loss_minority.png')
plt.clf()
plt.plot(acc_maj)
plt.savefig('acc_majority.png')
plt.clf()
plt.plot(acc_min)
plt.savefig('acc_minority.png')