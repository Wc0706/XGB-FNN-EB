## modules
import os
import os.path
import sys
import string
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import math
import dill
from optparse import OptionParser
import time
import re
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from collections import OrderedDict

USAGE = 'Usage: %prog [options] '
parser = OptionParser(USAGE, version='%prog 1.3')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device：{device}")

#############################
## parameters
#############################

# basic parameters

parser.add_option('-t', '--train_data', action='store', type="string",
                  default='', dest="tra_dat",
                  help='Input training data [required]')

parser.add_option('-v', '--valid_data', action='store', type="string",
                  default='', dest="val_dat",
                  help='Input validation data [required]')

parser.add_option('-p', '--predict_data', action='store', type="string",
                  default='',
                  dest="pre_dat",help='Input predict data [recommended]')

parser.add_option('-L', '--predict_data_list', action='store', type="string", default='', dest="pre_dat_list",
                  help='Input predict data list [recommended]')

parser.add_option('-l', '--label', action='store', type="string", default='Test1', dest="label",
                  help='Label of output files [default= %default]')

parser.add_option('-r', '--learn_rate', action='store', type="float", default=5e-4, dest="lr_rate",
                  help='Learning rate of the neural network  [default= %default]')

parser.add_option('-n', '--epoch_num', action='store', type="int", default=500, dest="epoch_num",
                  help='Epoch num of training  [default= %default]')

parser.add_option('-N', '--model_num', action='store', type="str", default='', dest="model_num",
                  help='Model num of training  [default= %default]')

parser.add_option('-o', '--out_dir', action='store', type="string", default=os.getcwd(), dest="out_dir",
                  help='Output dir [default= %default]')

# high-level parameters

parser.add_option('-I', '--in_size', action='store', type="int", default=6, dest="in_size",
                  help='Size of input layer of the neural network  [default= %default]')

parser.add_option('-H', '--hidden_size', action='store', type="int", default=20, dest="hid_size",
                  help='Size of output layer of the neural network  [default= %default]')

parser.add_option('-O', '--out_size', action='store', type="int", default=1, dest="out_size",
                  help='Size of output layer of the neural network  [default= %default]')

parser.add_option('-g', '--gamma', action='store', type="float", default=0.8, dest="gamma",
                  help='Multiplicative factor of learning rate decay  [default= %default]')

parser.add_option('-s', '--step_size', action='store', type="int", default=120, dest="step_size",
                  help='Period of learning rate decay  [default= %default]')

parser.add_option('-w', '--weight_initialize', action='store', type="int", default=0, dest="wei_init",
                  help='If or not initialize the weight of FNN model, \'1|0\' for \'Yes|No\'. [default= %default]')

parser.add_option('-W', '--Weight_xgb', action='store', type="str", default=None, dest="wei_xgb",
                  help='Initialized weight of FNN model according to XBG. [default= %default]')

opt, args = parser.parse_args()

# deal with input parameters
mtrain_dat = opt.tra_dat
mvalid_dat = opt.val_dat
mpred_dat = opt.pre_dat
mpred_dat_list = opt.pre_dat_list

mepoch_num = opt.epoch_num
mmodel_num = opt.model_num

mlearn_rate = opt.lr_rate
mgamma = opt.gamma
mstep_size = opt.step_size

mflag = opt.label
mout_dir = opt.out_dir

min_size = opt.in_size
mhid_size = opt.hid_size
mout_size = opt.out_size

mwei_init = opt.wei_init
mwei_xgb = opt.wei_xgb

mbatch_size = 1
mtloss_cutoff = 0.05
mvloss_cutoff = 0.05
meloss_cutoff = 0.05

mpatience_epoch = 30
mratio_tra_val = 0.005
mmax_loss = 30
mact_neu = None

# check input files
if os.path.exists(mtrain_dat):
    mtrain_dat = os.path.abspath(mtrain_dat)
else:
    print("\nPlease input training dataset!\n\nUse -h or --help for more help information\n")
    sys.exit(1)

if os.path.exists(mvalid_dat):
    mvalid_dat = os.path.abspath(mvalid_dat)
else:
    print("\nPlease input validation dataset!\n\nUse -h or --help for more help information\n")
    sys.exit(1)

# check output directory
if os.path.isdir(mout_dir):
    pass
else:
    os.mkdir(mout_dir)

mout_dir = os.path.abspath(mout_dir)

if mpred_dat:
    mpred_dat, mpred_dat_name = mpred_dat.split(',')
    mpred_dat = os.path.abspath(mpred_dat)
elif (opt.pre_dat_list):
    mpred_dat_list = os.path.abspath(mpred_dat_list)
else:
    mpred_dat = mvalid_dat

# output files
mprefix = mflag

torch.multiprocessing.set_sharing_strategy('file_system')


## class & function definition
class Get_dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataframe = []
        with open(dataset, 'r') as f:
            for line in f:
                aline = line.split()
                values = [float(val) for val in aline]
                self.dataframe.append(values)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        aline = self.dataframe[index]
        input_data = torch.Tensor(aline[:-1])
        out_data = aline[-1]
        return input_data, out_data


class NN_Model(torch.nn.Module):
    def __init__(self, insize, hidsize, outsize):
        super().__init__()
        self.main = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(insize, hidsize)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(hidsize, hidsize)),
            ('relu2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(hidsize, hidsize)),
            ('relu3', torch.nn.ReLU()),
            ('fc4', torch.nn.Linear(hidsize, outsize)),
        ])
        )

    def forward(self, input=[]):
        return self.main(input)


class model_pred():
    def __init__(self, dataloader):
        self.mpred_loss_avg = 0
        self.mpred_test_x = []
        self.mpred_test_y = []
        self.mpred_pred_y = []
        self.mpred_all = []
        self.mall_pred_loss = []
        for index, data in enumerate(dataloader):
            input_data, out_data = data
            input_data, out_data = input_data.to(device), out_data.to(device)
            out_data = out_data.float()
            pred2pred_y = nn_net(input_data)

            out_data = out_data.unsqueeze(0)
            predloss = loss_func(pred2pred_y, out_data)

            predloss_item = predloss.item()
            predloss_item = "%.6f" % (predloss_item)
            predloss_item = float(predloss_item)

            self.mpred_loss_avg += predloss_item
            self.mall_pred_loss.append(predloss.item())
            self.mpred_test_x.append(input_data[:, -1])
            self.mpred_test_y.append(out_data)
            pred2pred_y_value = pred2pred_y.data.detach()
            self.mpred_pred_y.append(pred2pred_y_value)

        self.mpred_loss_avg /= len(dataloader)
        self.mpred_test_x = torch.cat(self.mpred_test_x, dim=0).view(-1).cpu().numpy()
        self.mpred_test_y = torch.cat(self.mpred_test_y, dim=0).view(-1).cpu().numpy()
        self.mpred_pred_y = torch.cat(self.mpred_pred_y, dim=0).view(-1).cpu().numpy()
        self.mpred_all = np.column_stack((self.mpred_test_x, self.mpred_test_y, self.mpred_pred_y))
        self.mpred_all = pd.DataFrame(self.mpred_all, columns=['Temp', 'Exp', 'Fit'])

    def out(self):
        return self.mpred_all, self.mpred_loss_avg


class FNN_pred():
    def __init__(self, model_num, mepoch_num, train_loss_avg, valid_loss_avg, logfile, lf, df_loss):
        pred_loss_all = []
        exp_pred_all = pd.DataFrame(columns=['Temp', 'Exp'])
        self.tlogs = ''
        self.df_loss = df_loss
        for i in range(len(pred_data_loaders)):
            pred_loss_avg = 0
            data_name = pred_data_loaders[i, 0]
            pred_loader = pred_data_loaders[i, 1]
            exp_pred_val, pred_loss = model_pred(dataloader=pred_loader).out()
            pred_loss_avg = pred_loss
            pred_data_out = data_name + "pred.out.xls"
            pred_data_out = mdir + "." + data_name + ".pred.xls"
            col_name = 'Fit' + str(m)
            exp_pred_val.rename(columns={'Fit': col_name}, inplace=True)
            name_keys = pred_all_dict.keys()
            all_columns = exp_pred_all.columns.tolist()
            val_columns = exp_pred_val.columns.tolist()
            if (len(name_keys) != 0):
                if (data_name in name_keys):
                    exp_pred_all = pred_all_dict[data_name]
                    all_columns = exp_pred_all.columns.tolist()
                    val_columns = exp_pred_val.columns.tolist()
                    if (col_name in all_columns) and (col_name in val_columns):
                        exp_pred_all[col_name] = exp_pred_val[col_name]
                    else:
                        exp_pred_all = pd.merge(exp_pred_all, exp_pred_val, on=['Temp', 'Exp'], how='right')
                else:
                    exp_pred_all = exp_pred_val
                pred_all_dict[data_name] = exp_pred_all
            else:
                exp_pred_all = exp_pred_val
                pred_all_dict[data_name] = exp_pred_all

            pred_loss_all.append([data_name, pred_loss_avg])
            exp_pred_val2 = exp_pred_val
            exp_pred_val2.at[0, 'Delt'] = 0
            rownum = exp_pred_val2.shape[0]
            for i in range(rownum):
                exp_pred_val2.at[i, 'Delt'] = exp_pred_val2.at[i, col_name] - float(exp_pred_val2.at[i, 'Exp'])
            pred_file = mdir + "." + data_name + ".data.xls"
            exp_pred_val2.to_csv(pred_file, sep="\t", index=0)
            delt_data = exp_pred_val2['Delt'].apply(lambda x: float(x)).tolist()
            y_exp = exp_pred_val2['Exp']
            y_pred = exp_pred_val2[colname]
            RMS = getRMS(delt_data)
            rRMS = 100 * RMS / np.mean(y_exp)
            Devi = getDev(RMS)
            SD = np.std(delt_data, ddof=1)
            rSD = 100 * SD / np.mean(y_exp)
            R2_val = 1 - np.sum((y_exp - y_pred) ** 2) / np.sum((y_exp - np.mean(y_exp)) ** 2)
            tlog = "\n" + data_name + " :\n"
            tlog += "RMS : " + str(RMS) + "\n"
            tlog += "rRMS : " + str("%.6f" % (abs(rRMS))) + " %" + "\n"
            tlog += "Diviation : " + str(Devi) + " %" + "\n"
            tlog += "SD : " + str("%.6f" % (SD)) + "\n"
            tlog += "rSD : " + str("%.6f" % (abs(rSD))) + " %" + "\n"
            tlog += "R2 : " + str("%.6f" % (R2_val)) + "\n"
            self.tlogs += tlog
        pred_loss_all = np.array(pred_loss_all)
        pred_loss_avg = float(pred_loss_all[0, 1])
        lr_list = scheduler.get_last_lr()
        tmp_lr = lr_list[-1]
        alog = "epoch: %d/%d   train_loss: %.6f    valid_loss: %.6f" % (
        epoch + 1, mepoch_num, train_loss_avg, valid_loss_avg)
        self.log_losses = [train_loss_avg, valid_loss_avg]
        for k in range(len(pred_loss_all)):
            dname, dloss = pred_loss_all[k, :]
            self.log_losses.append(dname + "_pred_loss:" + dloss)
        for k in range(len(pred_loss_all)):
            dname, dloss = pred_loss_all[k, :]
            dloss = "%.6f" % (float(dloss))
            alog += "    " + dname + "_pred_loss: " + str(dloss)
        alog += "\n"
        logfile = logfile + alog
        lf.write(alog)
        tmp_loss = {'Epoch': epoch + 1, 'Train_loss': train_loss_avg, 'Valid_loss': valid_loss_avg,
                    'Pred_loss': pred_loss_avg, 'Learning_rate': tmp_lr}
        self.df_loss = pd.concat([self.df_loss, pd.DataFrame([tmp_loss])], ignore_index=True)
        mlast_log = "%d/%d,%.6f,%.6f,%.6f,%.6f" % (epoch + 1, mepoch_num, train_loss_avg, valid_loss_avg, tra_val_ratio,
                                                   pred_loss_avg)
        mlast_log = mlast_log.split(",")
        losslog.append(mlast_log)
        mlast_log.insert(0, model_num)
        self.meval = mlast_log

    def out(self):
        return self.meval, self.log_losses, self.tlogs, self.df_loss


def switch_flag(T):
    flag = ''
    if (T <= 1000):
        flag = 'Low'
    elif (1000 < T and T <= 1200):
        flag = 'Mid'
    elif (T > 1200):
        flag = 'High'
    return flag


def check_trend(x, y, z):
    flag = 1
    if (x < y and y < z):
        flag = 1
    elif (x > y and y > z):
        flag = 0
    elif (x >= 10 * y):
        flag = 0
    return flag


def accuracy_trend(all_valid_loss, mpatience_epoch):
    all_valid_loss = np.array(all_valid_loss)
    all_valid_losstr = all_valid_loss[::-1]
    all_valid_losstr = all_valid_losstr[:mpatience_epoch]
    trend_bin = math.floor(mpatience_epoch / 2)

    valid_loss_bin = all_valid_losstr[:2 * trend_bin]

    valid_loss_avg1 = all_valid_losstr[:trend_bin]
    valid_loss_avg1 = float(valid_loss_avg1.mean())
    valid_loss_avg2 = all_valid_losstr[trend_bin:2 * trend_bin]
    valid_loss_avg2 = float(valid_loss_avg2.mean())

    if valid_loss_avg1 >= valid_loss_avg2:
        losstrend = 0
    else:
        losstrend = 1

    return losstrend


def model_log(m, mlast_log):
    meval = []
    meval.append(m)
    meval.extend(mlast_log)
    return meval


def save_log(logfile, sprefix):
    logfile_name = sprefix + '.train.log'
    logfile_name = os.path.abspath(logfile_name)
    with open(logfile_name, 'w') as logf:
        logf.write(logfile)
    logf.close()


def getRMS(delt):
    rms = 0
    if len(delt) >= 1:
        for i in range(len(delt)):
            x = float(delt[i])
            rms = rms + math.pow(x, 2)
        rms = math.sqrt(rms / len(delt))
        rms = float("%.4f" % (rms))
    return rms


def getDev(rms):
    rms = float(rms)
    Dev = math.pow(10, rms)
    Dev = (Dev - 1) * 100
    Dev = float("%.1f" % (Dev))
    return (Dev)


def merge_pds(pd1, pd2, keys, how):
    if pd1 is None and pd2 is not None:
        pd1 = pd2
    else:
        pd1 = pd.merge(pd1, pd2, on=keys, how=how)

    return pd1


def get_sigma(data):
    n = len(data)
    avg = np.mean(data)
    sigma = 0
    if n >= 2:
        for i in range(n):
            sigma += math.pow(data[i] - avg, 2)
        sigma = math.sqrt(sigma / (n - 1))
    return (sigma)


# get train dataset
train_data = Get_dataset(dataset=mtrain_dat)
train_loader = DataLoader(train_data, batch_size=mbatch_size, shuffle=True, num_workers=0)
if len(train_loader) == 0:
    exit();

# valid data
valid_data = Get_dataset(dataset=mvalid_dat)
valid_loader = DataLoader(valid_data, batch_size=mbatch_size, shuffle=False, num_workers=0)
if len(valid_loader) == 0:
    exit();

# prediction data
pred_data_loaders = []
pred_data_names = []
pred_data_files = []

all_data_loaders = []
all_data_names = ['Train', 'Valid']
all_data_files = [mtrain_dat, mvalid_dat]

if mpred_dat_list:
    with open(mpred_dat_list, 'r') as f:
        for line in f:
            data_name, data_file = line.split()
            pred_data_names.append(data_name)
            if os.path.exists(data_file):
                pred_data_files.append(data_file)
                pred_data = Get_dataset(dataset=data_file)
                if len(pred_data) == 0:
                    exit()
                pred_loader = DataLoader(pred_data, batch_size=mbatch_size, shuffle=False,
                                         num_workers=0)
                pred_data_loaders.append([data_name, pred_loader])
                all_data_files.append(data_file)
    f.close()

if mpred_dat:
    pred_data = Get_dataset(dataset=mpred_dat)
    if len(pred_data) == 0:
        exit()
    pred_loader = DataLoader(pred_data, batch_size=mbatch_size, shuffle=False, num_workers=0)
    if mpred_dat_name is None:
        data_name = os.path.split(mpred_dat)[1]
    else:
        data_name = mpred_dat_name
    pred_data_loaders.append([data_name, pred_loader])
    pred_data_names.append(data_name)
    pred_data_files.append(mpred_dat)
    all_data_files.append(mpred_dat)

pred_data_loaders = np.array(pred_data_loaders)
all_data_names.extend(pred_data_names)


def save_weight(ts_weight, ts_bias, weight_filename):
    np_weight = ts_weight.detach().cpu().numpy()
    np_bias = ts_bias.detach().cpu().numpy().reshape(-1, 1)
    np_weight = ts_weight.detach().cpu().numpy()
    np_weight_bias = np.hstack((np_weight, np_bias))
    pd_weight_bias = pd.DataFrame(np_weight_bias)
    colnames = pd_weight_bias.columns.values.tolist()
    for i in range(len(colnames) - 1):
        colnames[i] = 'X' + str(colnames[i] + 1)
    colnames[-1] = 'bias'
    pd_weight_bias.columns = colnames
    rownames = pd_weight_bias.index.values.tolist()
    for i in range(len(rownames)):
        rownames[i] = 'w' + str(rownames[i] + 1)
    pd_weight_bias.index = rownames
    pd_weight_bias.to_csv(weight_filename, sep="\t", index=1, mode='w', header=True)


def weight_init(hid_neural, infeatures, fea_imp_ts, fan_mode):
    w = torch.Tensor(hid_neural, 1)
    kmu = torch.nn.init.kaiming_uniform_(w, mode=fan_mode, nonlinearity='relu')
    imp = fea_imp_ts
    imp_abs = abs(imp.numpy())
    imp_min = np.min(imp_abs)
    imp_max = np.max(imp_abs)

    imp_idx = np.argsort(imp_abs).tolist()[0]
    imp_secmin = 0
    imp_list = imp.numpy()
    for i in range(len(imp_idx) - 1):
        idx = imp_idx[i]
        if imp_list[0, idx] == 0 and imp_list[0, idx + 1] != 0:
            imp_secmin = imp_list[0, idx + 1]
            imp_list[0, idx] = np.random.uniform(0, 0.01 * imp_secmin, 1)

    imp = torch.Tensor(imp_list)

    imp_kmu = kmu * imp
    imp_kmu_sd = np.std(imp_kmu.numpy(), ddof=1)

    noise = np.random.uniform(-0.01 * imp_min, 0.01 * imp_min, hid_neural * infeatures).reshape(hid_neural, infeatures)
    pd_noise = pd.DataFrame(noise)
    pd_noise.to_csv("weigth.random.uniform.noise.xls", sep="\t", index=0)

    noise = torch.Tensor(noise)
    imp_kmu_noise = torch.nn.Parameter(imp_kmu + noise)
    return imp_kmu_noise


def hid_weight_init(in_neural, out_neural, fan_mode):
    hw = torch.Tensor(out_neural, in_neural)
    khw = torch.nn.init.kaiming_uniform_(hw, mode=fan_mode, nonlinearity='relu')
    khw = torch.nn.Parameter(khw)
    return (khw)


model_flag = 0
model_eval = []
total_log = ""
all_predvals = []

pred_all_dict = {}
allmodel2loss = {}
all_fit_dict = {}

msub_dir = mout_dir + '/1.FNN'

if os.path.isdir(msub_dir):
    pass
else:
    os.mkdir(msub_dir)

os.chdir(msub_dir)

patience = mpatience_epoch

m = mmodel_num

os.chdir(msub_dir)
stat_flag = 0
mdir = 'Model_' + str(m)
if not os.path.exists(mdir):
    os.mkdir(mdir)

absmdir = os.path.abspath(mdir)

print_msg = '\nModel_' + str(m) + ' strted training:\n'

os.chdir(absmdir)

logfile = ''
losslog = []

lf = open("all.detail.log", "w+")

while stat_flag == 0:

    nn_net = NN_Model(insize=min_size, hidsize=mhid_size, outsize=mout_size).to(device)

    fc1_weight = nn_net.main.fc1.weight

    fc1_bias = nn_net.main.fc1.bias

    save_weight(fc1_weight, fc1_bias, "original.fc1.weight.bias.txt")

    save_weight(nn_net.main.fc2.weight, nn_net.main.fc2.bias, "original.fc2.weight.bias.txt")
    save_weight(nn_net.main.fc3.weight, nn_net.main.fc3.bias, "original.fc3.weight.bias.txt")
    save_weight(nn_net.main.fc4.weight, nn_net.main.fc4.bias, "original.fc4.weight.bias.txt")

    if mwei_init:
        actneu = None
        if mwei_xgb is None:
            imp = torch.Tensor([[0.3097, -0.1165, -0.0739, -0.0311, 0.0213, 0.4475]])
        else:
            xgb_imp = []
            xgb_imps = mwei_xgb.split(",")
            for i in xgb_imps:
                i = re.sub(re.compile(r'\s+'), '', i)
                xgb_imp.append(float(i))
            xgb_imp = np.array(xgb_imp).reshape(1, -1)
            imp = torch.Tensor(xgb_imp)

        fc1_weight = weight_init(mhid_size, min_size, imp, 'fan_in')
        nn_net.main.fc1.weight = fc1_weight
        save_weight(fc1_weight, nn_net.main.fc1.bias, "initiated.fc1.weight.bias.txt")

    optimizer = torch.optim.Adam(nn_net.parameters(), lr=mlearn_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, mstep_size, gamma=mgamma, last_epoch=-1)

    loss_func = torch.nn.MSELoss(reduction='mean')

    df_loss = pd.DataFrame(columns=['Epoch', 'Train_loss', 'Valid_loss', 'Learning_rate'])
    all_train_loss = []
    all_valid_loss = []

    colname = 'Fit' + str(m)
    train_all = pd.DataFrame(columns=['Temp', 'Exp', colname])
    valid_all = pd.DataFrame(columns=['Temp', 'Exp', colname])

    for epoch in range(mepoch_num):

        train_loss_avg = 0
        train_test_x = []
        train_test_y = []
        train_pred_y = []

        for index, data in enumerate(train_loader):
            input_data, out_data = data
            input_data, out_data = input_data.to(device), out_data.to(device)
            out_data = out_data.float()
            nn_net.zero_grad()
            train2pred_y = nn_net(input_data)
            out_data = out_data.unsqueeze(0)
            trainloss = loss_func(train2pred_y, out_data)
            trainloss.backward()
            optimizer.step()
            train_loss_avg += trainloss.item()
            train_test_x.append(input_data[:, -1])
            train_test_y.append(out_data)
            train2pred_y_value = train2pred_y.data.detach()
            train_pred_y.append(train2pred_y_value)

        nn_net.eval()

        scheduler.step()
        train_loss_avg /= len(train_loader)
        all_train_loss.append(train_loss_avg)
        train_test_x = np.concatenate([x.cpu().numpy() for x in train_test_x])
        train_test_y = np.concatenate([y.cpu().numpy() for y in train_test_y])
        train_pred_y = np.concatenate([pred_y.cpu().numpy() for pred_y in train_pred_y])
        train_all = np.column_stack((train_test_x, train_test_y, train_pred_y))
        colname = 'Fit' + str(m)
        train_all = pd.DataFrame(np.asanyarray(train_all, dtype=float), columns=['Temp', 'Exp', colname])
        train_file = mdir + ".train.data.xls"

        train_all2 = train_all
        train_all2.at[0, 'Delt'] = 0
        rownum = train_all2.shape[0]
        for i in range(rownum):
            train_all2.at[i, 'Delt'] = train_all2.at[i, colname] - float(train_all2.at[i, 'Exp'])

        train_all2.to_csv(train_file, sep="\t", index=0)


        def model_stat(valid_all, colname, setname):
            if valid_all.shape[0] < 1:
                print("The data set is null!\n")
                sys.exit(0)

            delt_data = valid_all['Delt'].apply(lambda x: float(x)).tolist()
            y_exp = valid_all['Exp']
            y_pred = valid_all[colname]

            RMS = getRMS(delt_data)
            rRMS = 100 * RMS / np.mean(y_exp)
            Devi = getDev(RMS)
            SD = np.std(delt_data, ddof=1)
            rSD = 100 * SD / np.mean(y_exp)

            R2_val = 1 - np.sum((y_exp - y_pred) ** 2) / np.sum((y_exp - np.mean(y_exp)) ** 2)

            vlog = "\n" + setname + " :\n"
            vlog += "RMS : " + str(RMS) + "\n"
            vlog += "rRMS : " + str("%.6f" % (abs(rRMS))) + " %" + "\n"
            vlog += "Diviation : " + str(Devi) + " %" + "\n"
            vlog += "SD : " + str("%.6f" % (SD)) + "\n"
            vlog += "rSD : " + str("%.6f" % (abs(rSD))) + " %" + "\n"
            vlog += "R2 : " + str("%.6f" % (R2_val)) + "\n"

            return vlog, RMS


        trlog, RMS = model_stat(train_all2, colname, 'Train')

        valid_all, valid_loss_avg = model_pred(dataloader=valid_loader).out()
        colname = 'Fit' + str(m)
        valid_all.rename(columns={'Fit': colname}, inplace=True)

        valid_all2 = valid_all
        valid_all2.at[0, 'Delt'] = 0
        rownum = valid_all2.shape[0]
        for i in range(rownum):
            valid_all2.at[i, 'Delt'] = valid_all2.at[i, colname] - float(valid_all2.at[i, 'Exp'])

        valid_file = mdir + ".valid.data.xls"
        valid_all2.to_csv(valid_file, sep="\t", index=0)

        vlog, RMS = model_stat(valid_all2, colname, 'Valid')

        all_valid_loss.append(valid_loss_avg)

        if train_loss_avg >= mmax_loss or valid_loss_avg >= mmax_loss:
            lf.close()
            lf = open("all.detail.log", "w+")
            break

        tra_val_ratio = 0
        if valid_loss_avg != 0:
            tra_val_ratio = train_loss_avg / valid_loss_avg

        meval, log_losses, tlogs, df_loss = FNN_pred(m, mepoch_num, train_loss_avg, valid_loss_avg, logfile, lf,
                                                     df_loss).out()

        if tra_val_ratio <= mratio_tra_val or tra_val_ratio >= 1 / mratio_tra_val:
            stat = ['Overfit', 6]
            stat_flag = 0
            meval.extend(stat)
            model_eval.append(meval)
            log_losses.extend(stat)
            allmodel2loss[str(m)] = log_losses
            save_log(logfile, absmdir + '/Model_' + str(m))
            break

        if epoch + 1 >= mpatience_epoch and epoch + 1 >= int(mepoch_num * 0.1):
            acc_trend = accuracy_trend(all_valid_loss, mpatience_epoch)
            if acc_trend == 0 and train_loss_avg <= meloss_cutoff and valid_loss_avg <= meloss_cutoff:
                detail_log = "Warning[2]: Model training will be terminated with no validation loss decrease!\n"
                stat = ['Early_stopping', 2]
                stat_flag = 1
                meval.extend(stat)
                model_eval.append(meval)
                log_losses.extend(stat)
                allmodel2loss[str(m)] = log_losses
                lf.write(detail_log)
                break

        if train_loss_avg <= mtloss_cutoff and valid_loss_avg <= mvloss_cutoff:
            detail_log = "Warning[1]: Normal termination of training according to loss cutoff\n"
            model_flag = 1
            stat = ['Qualified', 1]
            stat_flag = 1
            meval.extend(stat)
            model_eval.append(meval)
            log_losses.extend(stat)
            allmodel2loss[str(m)] = log_losses
            torch.save(nn_net, mprefix + '.Model_' + str(m) + ".pth")
            lf.write(detail_log)
            break

        if (epoch == mepoch_num - 1) and (train_loss_avg <= meloss_cutoff and valid_loss_avg <= meloss_cutoff):
            detail_log = "Warning[3]: Training finished and satisfied to earlystopping loss cutoff.\n"
            model_flag = 1
            stat = ['Finished', 3]
            stat_flag = 1
            meval.extend(stat)
            model_eval.append(meval)
            log_losses.extend(stat)
            allmodel2loss[str(m)] = log_losses
            torch.save(nn_net, mprefix + '.Model_' + str(m) + ".pth")
            lf.write(detail_log)
            break

        if (epoch == mepoch_num - 1) and not (train_loss_avg <= meloss_cutoff and valid_loss_avg <= meloss_cutoff):
            detail_log = "Warning[4]: Training finished but did not satisfy to earlystopping loss cutoff!\n"
            model_flag = 1
            stat = ['Finished', 4]
            stat_flag = 1
            meval.extend(stat)
            model_eval.append(meval)
            log_losses.extend(stat)
            allmodel2loss[str(m)] = log_losses
            torch.save(nn_net, mprefix + '.Model_' + str(m) + ".pth")
            lf.write(detail_log)
            break

torch.save(nn_net, mprefix + '.trained' + ".model.pth")

all_keys = all_fit_dict.keys()

if 'Train' in all_keys:
    all_fit_dict['Train'] = merge_pds(all_fit_dict['Train'], train_all, ['Temp', 'Exp'], 'right')
else:
    all_fit_dict['Train'] = train_all

if 'Valid' in all_keys:
    all_fit_dict['Valid'] = merge_pds(all_fit_dict['Valid'], valid_all, ['Temp', 'Exp'], 'right')
else:
    all_fit_dict['Valid'] = valid_all

total_log = logfile

df_loss[['Epoch']] = df_loss[['Epoch']].astype(int)

torch.save(nn_net, mprefix + '.trained' + ".model.pth")

save_weight(nn_net.main.fc1.weight, nn_net.main.fc1.bias, "trained.fc1.weight.bias.txt")
save_weight(nn_net.main.fc2.weight, nn_net.main.fc2.bias, "trained.fc2.weight.bias.txt")
save_weight(nn_net.main.fc3.weight, nn_net.main.fc3.bias, "trained.fc3.weight.bias.txt")
save_weight(nn_net.main.fc4.weight, nn_net.main.fc4.bias, "trained.fc4.weight.bias.txt")

train_test_x = np.array(train_all)[:, 0]
train_test_y = np.array(train_all)[:, 1]
train_pred_y = np.array(train_all)[:, 2]

valid_test_x = np.array(valid_all)[:, 0]
valid_test_y = np.array(valid_all)[:, 1]
valid_pred_y = np.array(valid_all)[:, 2]

T_err = []
error_list = []
for i in range(len(valid_test_y)):
    error = (valid_test_y[i] - valid_pred_y[i]) / valid_test_y[i]
    error_list.append(error)

RMS_list = []
RMS = 0
for i in range(len(valid_test_y)):
    x = valid_test_y[i] - valid_pred_y[i]
    RMS_list.append(x)
    T_err.append([valid_test_x[i], error_list[i] * 100])
RMS = math.sqrt(sum([x ** 2 for x in RMS_list]) / len(RMS_list))
deviation = math.pow(10, RMS) - 1
devi = str("%.1f" % (deviation * 100)) + ' %'

T_err_sort = T_err
T_err_sort.sort(key=lambda x: x[0])

T_err_sort = np.array(T_err_sort)

T_err_data = np.empty(shape=[0, 3], dtype=complex)
for i in range(len(T_err_sort)):
    flag = switch_flag(T_err_sort[i, 0])
    line = np.append(T_err_sort[i], flag)
    T_err_data = np.row_stack((T_err_data, line))

Terro_df = pd.DataFrame(T_err_data)
Terro_df.columns = ['T', 'Erro', 'Group']
Terro_df[['T', 'Erro']] = Terro_df[['T', 'Erro']].astype(float)

import seaborn

plt.figure()
ax = seaborn.boxplot(x="Group", y="Erro", data=Terro_df)
ax = seaborn.swarmplot(x="Group", y="Erro", data=Terro_df, color=".25", size=3.5)

nobs = Terro_df['Group'].value_counts(sort=False).reindex(pd.unique(Terro_df['Group']))
nobs = [str(x) for x in nobs.tolist()]
nobs = [' ' * 15 + "n: " + i for i in nobs]

medians = Terro_df.groupby(['Group'])['Erro'].median().values

pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.05, nobs[tick], horizontalalignment='center', size='10', color='r')
plt.ylabel('Prediction erro (%)')
plt.savefig(mprefix + '.T_error.boxplot.jpg', dpi=600)
plt.savefig(mprefix + '.T_error.violinplot.eps', dpi=600, format='eps')
plt.close()

plt.figure()
pos = range(len(nobs))
ax2 = seaborn.violinplot(x="Group", y="Erro", data=Terro_df)

plt.ylabel('Prediction erro (%)')
plt.savefig(mprefix + '.T_error.violinplot.jpg', dpi=600)
plt.savefig(mprefix + '.T_error.violinplot.eps', dpi=600, format='eps')
plt.close()

total_log += trlog + "\n"
total_log += vlog + "\n"
total_log += tlogs + "\n"

logfile_name = mprefix + '.train.log'
with open(logfile_name, 'w') as logf:
    logf.write(total_log)
logf.close()

fig = plt.figure()
fig.set_size_inches(10, 12)

plt.subplot(2, 1, 1)
plt.plot(df_loss[['Epoch']], df_loss[['Train_loss']], 'r-', label='Train loss')
plt.title('Fitting loss of training/validation dataset')
plt.ylabel('training loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_loss[['Epoch']], df_loss[['Valid_loss']], 'k-', label='Valid loss')
plt.xlabel('Epoch number')
plt.ylabel('Validation loss')
plt.legend()

plt.savefig(mprefix + '.fit_loss.jpg', dpi=600)
plt.savefig(mprefix + '.fit_loss.eps', dpi=600, format='eps')

plt.close(fig)

tloss = []
vloss = []
np_loss = np.array(df_loss)
[tloss.append(math.log(x, 10)) for x in np_loss[:, 1]]
[vloss.append(math.log(x, 10)) for x in np_loss[:, 2]]

fig = plt.figure()
fig.set_size_inches(10, 12)

plt.subplot(2, 1, 1)
plt.plot(df_loss[['Epoch']], tloss, 'r-', label='Train loss')
plt.title('Fitting loss of training/validation dataset')
plt.ylabel('Training loss (log10)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_loss[['Epoch']], vloss, 'k-', label='Valid loss')
plt.xlabel('Epoch number')
plt.ylabel('Validation loss (log10)')
plt.legend()
plt.savefig(mprefix + '.fit_loss_log10.jpg', dpi=600)
plt.savefig(mprefix + '.fit_loss_log10.eps', dpi=600, format='eps')

plt.close(fig)

fig = plt.figure()

lr_list = df_loss['Learning_rate']
plt.plot(df_loss[['Epoch']], lr_list, 'r-', label='Learning rate')
plt.title('Learning rate of training/validation process')
plt.ylabel('Learning rate')
plt.legend()
plt.savefig(mprefix + '.epoch.lr.jpg', dpi=600)
plt.savefig(mprefix + '.epoch.lr.eps', dpi=600, format='eps')

plt.close(fig)

fig = plt.figure()
lr_log = []
lr_list = np.array(df_loss['Learning_rate'].astype(float))
[lr_log.append(math.log(x, 10)) for x in lr_list]
plt.plot(df_loss[['Epoch']], lr_log, 'r-', label='Learning rate')
plt.title('Learning rate of training/validation process')
plt.ylabel('Learning rate (log10)')
plt.legend()
plt.savefig(mprefix + '.epoch.lr.log10.jpg', dpi=600)
plt.savefig(mprefix + '.epoch.lr.log10.eps', dpi=600, format='eps')

plt.close(fig)

fig = plt.figure()
fig.set_size_inches(10, 12)
plt.subplot(2, 1, 1)
plt.plot(valid_test_x, valid_test_y, 'r.', label='original data')
plt.plot(valid_test_x, valid_pred_y, 'k.', label='predicted value')
plt.title('Validation result of ' + mflag, fontsize=14, fontweight='bold')
plt.xlabel('Temperature(K)')
plt.ylabel('Reaction Rate(log) (cm3/molecule s)')
text_x = max(valid_test_x) - (max(valid_test_x) - min(valid_test_x)) * 0.1
text_y = min(valid_test_y) + (max(valid_test_y) - min(valid_test_y)) * 0.1
plt.text(text_x, text_y, "Deviation= " + str(devi), ha='center', va='bottom', size=10.5, \
         family="fantasy", color="r", style="italic", weight="light", \
         bbox=dict(facecolor="white", edgecolor="r", alpha=0.7)
         )
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(valid_test_x, error_list, 'b.', label='error')
plt.title('Testset fitting error')
plt.xlabel('Temperature(K)')
plt.ylabel('Erro (%)')
plt.legend()


def to_percent(val, pos):
    return '%.2f%%' % (100 * val)


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.savefig(mprefix + '.fit_result_error.jpg', dpi=600)
plt.savefig(mprefix + '.fit_result_error.eps', dpi=600, format='eps')
plt.close(fig)

torch.save(nn_net, mprefix + '.all' + ".model.pth")