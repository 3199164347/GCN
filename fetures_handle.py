import pandas as pd
import numpy as np
import csv
import random
import time


if __name__ == "__main__":
    start_time = time.time()
    input_file = 'input_4w_3_modified.csv'  # 输入文件名
    output_file = 'attribute_output.csv'  # 输出文件名
    columns_to_keep = [47, 11, 48, 10, 58, 7, 20, 21, 8]  # 要保留的列索引，例如[0, 2]表示第1列和第3列

    with open(input_file, 'r', newline='') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        for row in reader:
            writer.writerow([row[i] for i in columns_to_keep])

    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间（单位为秒）
    print("程序运行时间：", execution_time)





