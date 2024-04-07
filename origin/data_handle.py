import os
import random

import pandas as pd
import glob
import pickle
import joblib
all_data={}


# 读取所有 CSV 文件
csv_files = glob.glob('*.csv')  # 搜索当前目录下的所有 .csv 文件

data_frames = []

# 逐个读取 CSV 文件并合并数据
for file in csv_files:
    df = pd.read_csv(file)
    data_frames.append(df)

combined_data = pd.concat(data_frames, ignore_index=True)
'''
, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min,
'''
# 选择特定的列
selected_columns = [' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',
                    ' Fwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Bwd Packet Length Max',' Protocol', ' Fwd Packet Length Max',
                    ' Fwd Packet Length Mean', ' Packet Length Std', ' Total Length of Bwd Packets', ' Packet Length Variance',
                    ' Avg Fwd Segment Size', ' Average Packet Size', ' Fwd Packet Length Std', 'Flow Bytes/s',
                    ' Flow Packets/s', ' Bwd Packet Length Min',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min',
                    'Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',
                    ' Bwd IAT Max',' Bwd IAT Min',' Source IP', ' Destination IP', ' Label']

selected_data = combined_data[selected_columns]

for i, data in selected_data.iterrows():
    label=data[selected_columns[-1]]
    if all_data.get(label, '') == '':
        all_data[label] = []
    all_data[label].append(data.to_list())
save_data = []
save_label = ['BENIGN', 'DoS Hulk', 'DDoS', 'PortScan', 'Bot', 'Web Attack   Brute Force'] #dict_keys(['BENIGN', 'Infiltration', 'Bot', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'Web Attack   Brute Force', 'Web Attack   XSS', 'Web Attack   Sql Injection', 'PortScan'])
for _la in save_label:
    # _la = _la.upper()
    a = all_data[_la]
    random.shuffle(a)
    if len(a) > 2000:
        a = a[:2000]
    save_data.append(a)

joblib.dump(save_data, './{}.pkl'.format('save_data_extend_features'))