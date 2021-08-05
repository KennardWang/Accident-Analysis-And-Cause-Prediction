
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_density(data, row_num, type):

    para = data[:, 1:2].ravel()
    plt.figure(figsize=(16, 9))
    plt.title('Density')

    if type:
        plt.xlabel('District')
    else:
        plt.xlabel('Time')

    for i in range(0, row_num):
        if data[i][0] == 1:
            plt.bar(x=i, height=data[i][2], width=0.7, alpha=0.8, color='red', label="Cause 1")
            plt.text(i, data[i][2] + 1, str(data[i][2]), ha="center", va="bottom")
            plt.xticks(np.arange(row_num), para)
        elif data[i][0] == 2:
            plt.bar(x=i, height=data[i][2], width=0.7, alpha=0.8, color='yellow', label="Cause 2")
            plt.text(i, data[i][2] + 1, str(data[i][2]), ha="center", va="bottom")
            plt.xticks(np.arange(row_num), para)
        elif data[i][0] == 3:
            plt.bar(x=i, height=data[i][2], width=0.7, alpha=0.8, color='blue', label="Cause 3")
            plt.text(i, data[i][2] + 1, str(data[i][2]), ha="center", va="bottom")
            plt.xticks(np.arange(row_num), para)
        elif data[i][0] == 4:
            plt.bar(x=i, height=data[i][2], width=0.7, alpha=0.8, color='green', label="Cause 4")
            plt.text(i, data[i][2] + 1, str(data[i][2]), ha="center", va="bottom")
            plt.xticks(np.arange(row_num), para)

    # remove repetition
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()

if __name__ == '__main__':

    data = np.array(pd.read_excel('density_WithDistrict.xls'))
    row_num = data.shape[0]
    #print(data)

    # draw
    plot_density(data, row_num, 1)
