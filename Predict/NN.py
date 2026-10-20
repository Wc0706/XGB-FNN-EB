import numpy
import numpy as np
import matplotlib
import pandas as pd
import csv
import datetime
from optparse import OptionParser
from pandas.core.frame import DataFrame
import os
from numpy import random

path = os.path.dirname(os.path.dirname(__file__))
USAGE = 'Usage: %prog [options] '
parser = OptionParser(USAGE, version='%prog 1.3')

parser.add_option('-t', '--train_data', default='', dest="tra_dat", type="string",
                  help='Input training data [required]')
opt, args = parser.parse_args()
train = opt.tra_dat

xx = pd.read_csv(train, header=None)
xx = pd.DataFrame(xx)
tmp = xx.iloc[:, 5]
t = xx.shape[0]

# ===== 3 models; arr1 has 3 columns =====
arr1 = random.random(size=(t, 3))
arr1 = pd.DataFrame(arr1)

# ===== Model IDs (neuron count may differ; determined by each weight file) =====
fnn = []

for j in range(0, 3):
    xx = pd.read_csv(train, header=None)
    xx = pd.DataFrame(xx)
    yy = []
    for i in range(xx.shape[0]):
        x = xx.iloc[i, :]
        x = pd.DataFrame(x).T

        qq = str(fnn[j])

        # ---------- fc1 ----------
        data1 = pd.read_csv(path + '/predict/' + qq + '.fnn/testR1/1.FNN/Model_1/trained.fc1.weight.bias.txt',
                            header=0, sep='\t')
        df1 = pd.DataFrame(data1)
        aa = df1.iloc[:, -1]          # bias = last column
        df1 = df1.iloc[:, 1:-1]       # drop col 0 (index) and last col (bias)
        a = df1.T
        y = pd.DataFrame(np.dot(x, a))
        aa = pd.DataFrame(aa).reset_index(drop=True)
        for i2 in range(a.shape[1]):
            y.iloc[0, i2] = max(0, y.iloc[0, i2] + aa.iloc[i2, 0])

        # ---------- fc2 ----------
        data2 = pd.read_csv(path + '/predict/' + qq + '.fnn/testR1/1.FNN/Model_1/trained.fc2.weight.bias.txt',
                            header=0, sep='\t')
        df2 = pd.DataFrame(data2)
        bb = df2.iloc[:, -1]
        df2 = df2.iloc[:, 1:-1]
        b = df2.T
        y1 = pd.DataFrame(np.dot(y, b))
        bb = pd.DataFrame(bb).reset_index(drop=True)
        for i2 in range(b.shape[1]):
            y1.iloc[0, i2] = max(0, y1.iloc[0, i2] + bb.iloc[i2, 0])

        # ---------- fc3 ----------
        data3 = pd.read_csv(path + '/predict/' + qq + '.fnn/testR1/1.FNN/Model_1/trained.fc3.weight.bias.txt',
                            header=0, sep='\t')
        df3 = pd.DataFrame(data3)
        cc = df3.iloc[:, -1]
        df3 = df3.iloc[:, 1:-1]
        c = df3.T
        y2 = pd.DataFrame(np.dot(y1, c))
        cc = pd.DataFrame(cc).reset_index(drop=True)
        for i2 in range(c.shape[1]):
            y2.iloc[0, i2] = max(0, y2.iloc[0, i2] + cc.iloc[i2, 0])

        # ---------- fc4 (output layer, no ReLU) ----------
        data4 = pd.read_csv(path + '/predict/' + qq + '.fnn/testR1/1.FNN/Model_1/trained.fc4.weight.bias.txt',
                            header=0, sep='\t')
        df4 = pd.DataFrame(data4)
        dd = df4.iloc[:, -1]
        df4 = df4.iloc[:, 1:-1]
        d = df4.T
        y3 = pd.DataFrame(np.dot(y2, d))
        dd = pd.DataFrame(dd).reset_index(drop=True)
        for i2 in range(d.shape[1]):
            y3.iloc[0, i2] = y3.iloc[0, i2] + dd.iloc[i2, 0]

        yy.append(y3.values.flatten())

    yy = pd.DataFrame(yy)
    arr1.iloc[:, j] = yy.iloc[:, 0]

# ===== Average over the 3 models =====
arr2 = random.random(size=(t, 1))
arr2 = pd.DataFrame(arr2)
for i in range(len(arr2)):
    arr2.iloc[i, 0] = (arr1.iloc[i, 0] + arr1.iloc[i, 1] + arr1.iloc[i, 2]) / 3

t = pd.concat([tmp, arr2], axis=1)
t.columns = ['Temp', 'Fit']

c5_folder = os.path.join(path, "")
if not os.path.exists(c5_folder):
    os.makedirs(c5_folder)

txt_filename = ""
txt_save_path = os.path.join(c5_folder, txt_filename)

t.to_csv(txt_save_path, sep='\t', encoding='utf-8', index=None, header=True)
print(f"Results saved to: {txt_save_path}")