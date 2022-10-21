import torch
import numpy as np
from sklearn.metrics import mean_squared_error


# def my_metric(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)
#
# def my_metric2(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def RMSE(y_predict,y_test):
    return np.sqrt(mean_squared_error(y_predict,y_test))

def NRMSE(y_predict,y_test):
    return np.sqrt(mean_squared_error(y_predict,y_test))/(np.max(y_test)-np.min(y_test))

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:

    """
    with torch.no_grad():
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
    return r_val


def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    # Example:
    #     >>> x = np.random.randn(5,120)
    #     # result is a (5,5) matrix of correlations between rows
    #     >>> np_corr = np.corrcoef(x)
    #     >>> th_corr = corrcoef(torch.from_numpy(x))
    #     >>> np.allclose(np_corr, th_corr.numpy())
    #     # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

# def batch_pearsonr(x,y):
#     with torch.no_grad():
#         batch_size = x.size(0)
#         channel_num = x.size(-1)
#         # result = torch.reshape(torch.FloatTensor([pearsonr(x[batch,:,channel],y[batch,:,channel]) for batch in range(batch_size) for channel in range(channel_num)]),[batch_size,channel_num])
#         x_origin = []
#         for n in range(channel_num):
#             for i in range(batch_size):
#                 if i == 0:
#                     tmpchanneldata = x[i, :, n]
#                 else:
#                     tmpchanneldata = torch.cat((tmpchanneldata, x[i, 0:, n]),0)
#             x_origin.append(tmpchanneldata)
#         x_origin = torch.stack(x_origin,dim=1)
#
#         y_origin = []
#         for n in range(channel_num):
#             for i in range(batch_size):
#                 if i == 0:
#                     tmpchanneldata = y[i, :, n]
#                 else:
#                     tmpchanneldata = torch.cat((tmpchanneldata, y[i, 0:, n]),0)
#             y_origin.append(tmpchanneldata)
#         y_origin = torch.stack(y_origin,dim=1)
#     return torch.mean(torch.stack((pearsonr(x_origin[0],y_origin[0]), pearsonr(x_origin[1],y_origin[1]), pearsonr(x_origin[2],y_origin[2])), dim=0))

def channel1_batch_pearsonr(x,y):
    with torch.no_grad():
        batch_size = x.size(0)
        x_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = x[i, :, 0]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, x[i, 0:, 0]),0)
        x_origin.append(tmpchanneldata)
        x_origin = torch.stack(x_origin,dim=1)

        y_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = y[i, :, 0]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, y[i, 0:, 0]),0)
        y_origin.append(tmpchanneldata)
        y_origin = torch.stack(y_origin,dim=1)
    return pearsonr(torch.squeeze(x_origin),torch.squeeze(y_origin))

def channel2_batch_pearsonr(x,y):
    with torch.no_grad():
        batch_size = x.size(0)
        x_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = x[i, :, 1]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, x[i, 0:, 1]),0)
        x_origin.append(tmpchanneldata)
        x_origin = torch.stack(x_origin,dim=1)

        y_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = y[i, :, 1]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, y[i, 0:, 1]),0)
        y_origin.append(tmpchanneldata)
        y_origin = torch.stack(y_origin,dim=1)
    return pearsonr(torch.squeeze(x_origin),torch.squeeze(y_origin))

def channel3_batch_pearsonr(x,y):
    with torch.no_grad():
        batch_size = x.size(0)
        x_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = x[i, :, 2]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, x[i, 0:, 2]),0)
        x_origin.append(tmpchanneldata)
        x_origin = torch.stack(x_origin,dim=1)

        y_origin = []
        for i in range(batch_size):
            if i == 0:
                tmpchanneldata = y[i, :, 2]
            else:
                tmpchanneldata = torch.cat((tmpchanneldata, y[i, 0:, 2]),0)
        y_origin.append(tmpchanneldata)
        y_origin = torch.stack(y_origin,dim=1)
    return pearsonr(torch.squeeze(x_origin),torch.squeeze(y_origin))

if __name__ == '__main__':
    import scipy
    import numpy as np
    x = np.random.randn(100)
    y = np.random.randn(100)
    sp_corr = np.corrcoef(x,y)[0,1]
    th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
    print(sp_corr)
    print('-----'*10)
    print(th_corr)

    np.allclose(sp_corr, th_corr)