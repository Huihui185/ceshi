# import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch import optim
# from data_loader_feature import load_training,load_testing
from torch.autograd import Variable
import os
import metric
from ninapro_12_200 import NINAPRO_1
from conv1d_gru import CONVGRUNet
# convlstm块
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score
import numpy as np
import time
from tensorboardX import SummaryWriter

# from revin_norm import RevIN


def RMSE(y_test,y_predict):
    rmse = np.sqrt(mean_squared_error(y_test,y_predict))
    return rmse
def NRMSE(y_predict,y_test):
    nrmse =  np.sqrt(mean_squared_error(y_predict,y_test))/(np.max(y_test)-np.min(y_test))
    return nrmse

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_cc = []
# subject = [1,5,13,14,17,19,23,33]
for i in range(40):
# for i in subject:
    LR = 0.001
    EPOCH = 200
    min_rmse = 0
    min_nrmse = 0
    BATCH_SIZE = 1 * 64
    OUTPUT = 12
    INPUT = 12
    TIME_STEP = 200
    WINDOW_SIZE = 200
    # convlstm = CONVLSTM(1,[64,32,10],(3,3),3,True)   # input_dim  , hidden_dims ,ker_size, numlayers,
    loss_fn = torch.nn.MSELoss()

    model = CONVGRUNet(200, INPUT, OUTPUT, [64, 32, 12], 11, 3, True)
    model = model.to(device)

    obj = i+1
    print('s:', obj)
    max_test = 0
    max_epoch = 0
    max_r2 = 0

    writer = SummaryWriter(
                'model_save/200_1/conv1dgru/runs/Sub' + str(obj) + str(time.strftime("_%m-%d-%H-%M", time.localtime())))

    # DB2 200_1

    data_name = 'rms_emg'

    data_read = NINAPRO_1('data/'+data_name+'/train_data' + str(obj) + '.csv',
                          'data/glove/train_glove' + str(obj) + '.csv',
                          'data/'+data_name+'/test_data' + str(obj) + '.csv',
                          'data/glove/test_glove' + str(obj) + '.csv',TIME_STEP,WINDOW_SIZE,train=True)
    data_read_test = NINAPRO_1('data/'+data_name+'/train_data' + str(obj) + '.csv',
                          'data/glove/train_glove' + str(obj) + '.csv',
                          'data/'+data_name+'/test_data' + str(obj) + '.csv',
                          'data/glove/test_glove' + str(obj) + '.csv',TIME_STEP,WINDOW_SIZE,train=False)

    train_loader = DataLoader(data_read, BATCH_SIZE, False, num_workers=0, drop_last=True)
    test_loader = DataLoader(data_read_test, BATCH_SIZE, False, num_workers=0, drop_last=True)
    # indices = torch.LongTensor([0, 2, 3, 4, 5, 7, 11, 14, 15, 21])
    indices = torch.LongTensor([0,1, 2, 4, 5, 7, 8, 11, 12, 15, 16,21])
    # glove_max = np.array(torch.index_select(glove_max, 0, indices))
    # glove_min = np.array(torch.index_select(glove_min, 0, indices))
    # glove_max = np.array(glove_max)
    # glove_min = np.array(glove_min)

    print('aaaaaaaaaaaaa')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    epoch_start_time = time.time()
    hidden = None
    for epoch in range(EPOCH):
        total_output = torch.Tensor([]).to(device)
        total_target = torch.Tensor([]).to(device)
        total_output_test = torch.Tensor([]).to(device)
        total_target_test = torch.Tensor([]).to(device)
        model.train()

        for idx, (data, target) in enumerate(train_loader):

            # data = torch.Tensor(np.transpose(np.array(data, dtype='float32'), (0, 2, 1)))
            # data = data.reshape(BATCH_SIZE,1, INPUT, 200).to(device)
            data = data.to(device)
            target = torch.Tensor(
                np.transpose(np.array(target, dtype='float32'), (0, 1, 2)))  # (batchsize,inputsize,channel,)

            # indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            target = torch.index_select(target[:, 99:100, :], 2, indices)
            target = target.squeeze(1).to(device)
            # target = target.reshape(-1,1,1,28,28)
            pred= model(data)
            # for l in range(len(hidden)):
                # hidden[l].detach_()
            # total_output_test = torch.cat([total_output_test, pred.view([64, 10])])
            # total_target_test = torch.cat([total_target_test, target.view([64, 10])])
            # pdb.set_trace()
            loss = loss_fn(pred, target)
            total_output = torch.cat([total_output, pred.view(BATCH_SIZE, OUTPUT)])
            total_target = torch.cat([total_target, target.view(BATCH_SIZE, OUTPUT)])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        writer.add_scalar('loss', loss, epoch)
            # print('epoch: {}, loss: {}'.format(epoch, loss.item()))
        model.eval()
        with torch.no_grad():
            hidden1 = None
            for step, (data, target) in enumerate(test_loader):
                # x = torch.Tensor(
                    # np.transpose(np.array(data, dtype='float32'), (0, 2, 1)))  # (batchsize,channel,inputsize)
                # x = x.reshape(BATCH_SIZE, 1, INPUT, 200).to(device)
                x = data.to(device)
                y = torch.Tensor(
                    np.transpose(np.array(target, dtype='float32'), (0, 1, 2))).to(device)  # (batchsize,inputsize,channel,)
                # indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])

                y = torch.index_select(y[:, 99:100, :], 2, indices.to(device))
                y = y.squeeze(1).to(device)
                output_test= model(x)
                # hidden1 = hidden1[0].detach_()
                total_output_test = torch.cat([total_output_test, output_test.view([BATCH_SIZE, OUTPUT])])
                total_target_test = torch.cat([total_target_test, y.view([BATCH_SIZE, OUTPUT])])

            aver_test = 0
            aver_train = 0
            aver_r2_test = 0
            aver_rmse_test = 0
            aver_nrmse_test = 0

            # 真是角度的最大最小值，用于反归一化

            for id in range(OUTPUT):
                cc_train = metric.pearsonr(total_output[:, id], total_target[:, id])  # 计算特征与目标变量之间的相关度
                cc_test = metric.pearsonr(total_output_test[:, id], total_target_test[:, id])
                r2_test = r2_score(total_target_test[:, id].cpu(),total_output_test[:, id].cpu())
                writer.add_scalar('train/Accurancy' + str(id), cc_train, epoch)
                writer.add_scalar('test/Accurancy' + str(id), cc_test, epoch)
                pre = np.array(total_output_test[:, id].cpu())
                target = np.array(total_target_test[:, id].cpu())
                # 对真实角度进行还原，计算rms和 nrmse 都需要真实的角度进行计算
                # pre_glove = pre * (glove_max[id] - glove_min[id]) + glove_min[id]
                # true_glove = target * (glove_max[id] - glove_min[id]) + glove_min[id]

                rmse_test = RMSE(pre, target)
                nrmse_test = NRMSE(pre, target)

                aver_test = aver_test + cc_test
                aver_train = aver_train + cc_train
                aver_r2_test = aver_r2_test+ r2_test
                aver_rmse_test = aver_rmse_test + rmse_test
                aver_nrmse_test = aver_nrmse_test + nrmse_test
                # print('test/Accurancy' + str(id), cc_test, epoch)
            aver_test = aver_test / OUTPUT
            aver_train = aver_train / OUTPUT
            aver_r2_test = aver_r2_test / OUTPUT
            aver_rmse_test = aver_rmse_test / OUTPUT
            aver_nrmse_test = aver_nrmse_test / OUTPUT

            writer.add_scalar('test/tolacc', aver_test, epoch)
            writer.add_scalar('train/tolacc', aver_train, epoch)
            writer.add_scalar('test/tol_rmse', aver_rmse_test, epoch)
            writer.add_scalar('test/tol_nrmse', aver_nrmse_test, epoch)
            # print('train/Accurancy' + str(id), cc_train, epoch)

            print('epoch: ', epoch, 'loss: ', loss.item(), 'aver_train/accuracy: ', aver_train)
            print('epoch: ', epoch, '  |Ave/rmse: ----', aver_rmse_test, '|   nrmse: ', aver_nrmse_test,
                  ' | ave/Accurancy: ---', aver_test,'! r2: ',aver_r2_test)
            if aver_test > max_test:
                # torch.save(model, '/home/ZPH/deepnet/code/实验/model_save/convgru_shi/gru_S' + str(obj) + '.pkl')
                # torch.save(model, '/home/ZPH/deepnet/code/实验/model_save/200_1/conv1dgru/gru_S' + str(obj) + '.pkl')
                max_test = aver_test
                max_epoch = epoch
                max_r2 = aver_r2_test
                min_rmse = aver_rmse_test
                min_nrmse = aver_nrmse_test
            print('best_epoch: ', max_epoch, '  |MIN/rmse: ----', min_rmse, '|  MIN nrmse: ', min_nrmse,
                  ' | MAX/Accurancy: ---', max_test,' | MAX/R2: ---', max_r2)
    cost_time = time.time() - epoch_start_time
    aver_time = cost_time / EPOCH
    writer.add_scalar('cost_time', cost_time)
    writer.add_scalar('aver_time', aver_time)
    writer.add_scalar('min_rmse', min_rmse)
    writer.add_scalar('min_nrmse', min_nrmse)
    writer.add_scalar('max_test', max_test)
    writer.add_scalar('max_epoch', max_epoch)
    max_cc.append(max_test.cpu())
    print("cost_time: ", cost_time, "    |ave_time: ", aver_time)
max_cc = np.array(max_cc)
max_cc = pd.DataFrame(max_cc)
# max_cc.to_csv('result/u_norm_pca_convgru_6feature_max_cc.csv')
# max_cc.to_csv('result/norm_pca_convgru_6feature_max_cc.csv')
# max_cc.to_csv('result/norm_convgru_6feature_max_cc.csv')
# max_cc.to_csv('result/标准化_convgru1_6feature_max_cc.csv')
max_cc.to_csv('result/'+str(BATCH_SIZE)+'_norm_convgru1_rms_max_cc.csv')