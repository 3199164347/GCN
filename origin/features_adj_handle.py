import random
import torch

import joblib

save_data = joblib.load('./{}.pkl'.format('save_data_extend_features'))

all_host_to_host = []
_id = 0
save_label = {'BENIGN':0, 'DoS Hulk':1, 'DDoS':2, 'PortScan':3, 'Bot':4, 'Web Attack   Brute Force':5}
all_data = []
for _data in save_data:
    all_data+=_data
flow_host_id={}
random.shuffle(all_data)
format_data = []
for i, item in enumerate(all_data):
    feature = item[:-3]
    all_host_to_host.append([item[-3], item[-2]])

    v = [item[-3], item[-2]]
    l = save_label[item[-1]]
    format_data.append([feature, v, l])

features = torch.zeros((len(format_data), 34))
adj_m = torch.zeros((len(format_data), len(format_data)))
label=[]
for i, item in enumerate(format_data):
    features[i] = torch.tensor(item[0])
    for h in item[1]:
        for j, host in enumerate(all_host_to_host):
            if h in host and adj_m[i][j] == 0:
                adj_m[i][j] = 1
                adj_m[j][j] = 1
    label.append(item[2])

torch.save(features, './features_extend.pt')
torch.save(adj_m, './adj_extend.pt')
joblib.dump(label, './{}.pkl'.format('label_extend'))
