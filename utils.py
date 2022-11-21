import torch
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

def obtain_source_like_data_index(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        # sel_path = iter_test.dataset.imgs
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    #模型预测的结果
    con, predict = torch.max(all_output, 1)
    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    
    #寻找每一个样本的邻居
    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K)
    #这里的邻居包括自己
    near_label = torch.tensor(pred_label)[idx_near]

    #当前样本和邻居之间的相似度
    dis_near = dis_near[:,1:]
    neigh_dis = []
    for index in range(len(pred_label)):
        neigh_dis.append(np.mean(np.array(dis_near[index])))
    neigh_dis = np.array(neigh_dis)
    #构造邻居标签概率分布 pro_cls_near
    pro_clu_near = []
    for index in range(len(near_label)):
        #把每一个样本的概率值存起来
        label = np.zeros(args.class_num)
        for cls in range(args.class_num):
            cls_filter = (near_label[index] == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            pro = len(list_loc)/len(near_label[index])
            label[cls] = pro
        pro_clu_near.append(label)

    pro_clu_near = torch.tensor(pro_clu_near)
    #利用簇的熵来表示簇的不确定程度
    ent = torch.sum(- pro_clu_near  * torch.log(pro_clu_near + args.epsilon), dim=1)
    ent = ent.float()
    closeness = torch.tensor(neigh_dis)
    #stand = ent
    stand = ent * closeness
    
    sl_index = []#collect the labels of source-like-data
    # true_label = []
    sor = np.argsort(stand)
    index = 0
    #选择比例 ratio
    args.SSN =  int(len(pred_label) * args.ratio)
    sl_index = sor[0: args.SSN]



    #sel_pred_label
    pred_sel_label = pred_label[sl_index]
    #sel_true_label
    true_sel_label = all_label[sl_index]
    #挑选样本的准确率
    acc_sel  = np.sum(pred_sel_label == true_sel_label.float().numpy()) / len(true_sel_label)


    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc_sel * 100)
    print(log_str +'\n')
    
    return sl_index, pred_sel_label


def obtain_source_like_data_index_alpha(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        # sel_path = iter_test.dataset.imgs
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    #模型预测的结果
    con, predict = torch.max(all_output, 1)
    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    
    #寻找每一个样本的邻居
    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K)
    #这里的邻居包括自己
    near_label = torch.tensor(pred_label)[idx_near]

    #当前样本和邻居之间的相似度
    dis_near = dis_near[:,1:]
    neigh_dis = []
    for index in range(len(pred_label)):
        neigh_dis.append(np.mean(np.array(dis_near[index])))
    neigh_dis = np.array(neigh_dis)
    #构造邻居标签概率分布 pro_cls_near
    pro_clu_near = []
    for index in range(len(near_label)):
        #把每一个样本的概率值存起来
        label = np.zeros(args.class_num)
        for cls in range(args.class_num):
            cls_filter = (near_label[index] == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            pro = len(list_loc)/len(near_label[index])
            label[cls] = pro
        pro_clu_near.append(label)

    pro_clu_near = torch.tensor(pro_clu_near)
    #利用簇的熵来表示簇的不确定程度
    ent = torch.sum( - pro_clu_near  * torch.log( pro_clu_near + args.epsilon), dim=1)
    ent = ent.float()
    print(ent, ent.min(), ent.max())
    closeness = torch.tensor(neigh_dis)
    print(closeness, closeness.min(), closeness.max())
    #stand = ent
    stand = ent * closeness #neighbor uncertainty
    
    sl_index = []#collect the index of source-like-data
    # true_label = []
    sor = np.argsort(stand)
    index = 0
    #选择比例 ratio
    args.SSN =  int(len(pred_label) * args.ratio)
    sl_index = sor[0: args.SSN]

    #sel_pred_label
    pred_sel_label = pred_label[sl_index]
    #sel_true_label
    true_sel_label = all_label[sl_index]
    #挑选样本的准确率
    acc_sel  = np.sum(pred_sel_label == true_sel_label.float().numpy()) / len(true_sel_label)


    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc_sel * 100)
    print(log_str)

    #TODO compute alpha
    #alpha = distribution distance * target data neighbor uncertainty
    tl_index = []
    tl_index = sor[-args.SSN : -1]#make sure the amount of 

    #print(stand)
    standnumpy = stand.cpu().numpy()
    mean_neighbor_uncer = np.mean(standnumpy[tl_index])

    sl_feature, tl_feature = all_fea[sl_index], all_fea[tl_index]

    disc = mmd_rbf(sl_feature, tl_feature, gamma=1.0)
    #disc = mmd_linear(sl_feature, tl_feature)

    alpha = (1-disc)*(1-mean_neighbor_uncer)

    print('alpha {:.2f} = (1-disc) {:.5f} * (1-uncertainty) {:.2f}'.format(alpha, 1-disc, 1-mean_neighbor_uncer) + '\n')
    
    return sl_index, pred_sel_label, all_label, alpha, stand, tl_index



def obtain_domain_prior(all_fea, all_output, args):
    #模型预测的结果
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    #accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    #寻找每一个样本的邻居
    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=20)
    #这里的邻居包括自己
    near_label = torch.tensor(pred_label)[idx_near]

    #当前样本和邻居之间的相似度
    dis_near = dis_near[:,1:]
    neigh_dis = []
    for index in range(len(pred_label)):
        neigh_dis.append(np.mean(np.array(dis_near[index])))
    neigh_dis = np.array(neigh_dis)
    #print(neigh_dis)
    #构造邻居标签概率分布 pro_cls_near
    pro_clu_near = []
    for index in range(len(near_label)):
        #把每一个样本的概率值存起来
        label = np.zeros(args.class_num)
        for cls in range(args.class_num):
            cls_filter = (near_label[index] == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            pro = len(list_loc)/len(near_label[index])
            label[cls] = pro
        pro_clu_near.append(label)

    pro_clu_near = torch.tensor(pro_clu_near)
    #利用簇的熵来表示簇的不确定程度
    ent = torch.sum(- pro_clu_near * torch.log(pro_clu_near + args.epsilon), dim=1)
    ent = ent.float()
    #print(ent, ent.min(), ent.max())
    closeness = torch.tensor(neigh_dis)

    alpha = closeness.mean()/ent.mean()
    #alpha = (1-disc)*(1-mean_neighbor_uncer)

    #print('alpha {:.2f} = closeness {:.5f} / ent {:.5f}'.format(alpha, closeness.mean(), ent.mean()) + '\n')
    
    return alpha


def obtain_trust_data_index(all_fea, all_output, all_label, args):
    '''
    obtain the source-like and trusted data, updating the new reliable cluster centers.
    '''
    all_output = nn.Softmax(dim=1)(all_output)
    #模型预测的结果
    con, predict = torch.max(all_output, 1)
    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    
    #寻找每一个样本的邻居
    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K)
    #这里的邻居包括自己
    near_label = torch.tensor(pred_label)[idx_near]

    #当前样本和邻居之间的相似度
    dis_near = dis_near[:,1:]
    neigh_dis = []
    for index in range(len(pred_label)):
        neigh_dis.append(np.mean(np.array(dis_near[index])))
    neigh_dis = np.array(neigh_dis)
    #构造邻居标签概率分布 pro_cls_near
    pro_clu_near = []
    for index in range(len(near_label)):
        #把每一个样本的概率值存起来
        label = np.zeros(args.class_num)
        for cls in range(args.class_num):
            cls_filter = (near_label[index] == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            pro = len(list_loc)/len(near_label[index])
            label[cls] = pro
        pro_clu_near.append(label)

    pro_clu_near = torch.tensor(pro_clu_near)
    #利用簇的熵来表示簇的不确定程度
    ent = torch.sum( - pro_clu_near  * torch.log( pro_clu_near + args.epsilon), dim=1)
    ent = ent.float()

    closeness = torch.tensor(neigh_dis)
    #stand = ent
    stand = ent * closeness #neighbor uncertainty
    
    # sl_index = []#collect the index of source-like-data
    # # true_label = []
    # sor = np.argsort(stand)
    # index = 0
    # #选择比例 ratio
    # args.SSN =  int(len(pred_label) * args.ratio)
    # sl_index = sor[0: args.SSN]

    # #sel_pred_label
    # pred_sel_label = pred_label[sl_index]
    # #sel_true_label
    # true_sel_label = all_label[sl_index]
    # #挑选样本的准确率
    # acc_sel  = np.sum(pred_sel_label == true_sel_label.float().numpy()) / len(true_sel_label)

    # log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc_sel * 100)
    # print(log_str)

    # #alpha = distribution distance * target data neighbor uncertainty
    # tl_index = []
    # tl_index = sor[-args.SSN : -1]#make sure the amount of 

    # #print(stand)
    # standnumpy = stand.cpu().numpy()
    # mean_neighbor_uncer = np.mean(standnumpy[tl_index])

    # sl_feature, tl_feature = all_fea[sl_index], all_fea[tl_index]

    # disc = mmd_rbf(sl_feature, tl_feature, gamma=1.0)
    # #disc = mmd_linear(sl_feature, tl_feature)

    # alpha = (1-disc)*(1-mean_neighbor_uncer)

    # print('alpha {:.2f} = (1-disc) {:.5f} * (1-uncertainty) {:.2f}'.format(alpha, 1-disc, 1-mean_neighbor_uncer) + '\n')
    
    return stand

# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


if __name__ == '__main__':
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print(a)
    print(b)
    print(mmd_linear(a, b))  # 6.0
    print(mmd_rbf(a, b))  # 0.5822
    print(mmd_poly(a, b))  # 2436.5