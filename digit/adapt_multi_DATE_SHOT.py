import argparse
import os, sys
import os.path as osp
from xml import dom
import torchvision
import numpy as np
import torch
import time
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import DigitFiveList
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import scipy.io as sio

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    transform_dig=transforms.Compose([
                    transforms.Resize(28),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
    dsets["target"] = DigitFiveList(args, args.t_dset_path, transform=transform_dig)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["target_"] = DigitFiveList(args, args.t_dset_path, transform=transform_dig)
    dset_loaders["target_"] = DataLoader(dsets["target_"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["test"] = DigitFiveList(args, args.test_dset_path, transform=transform_dig)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF_list = [network.LeNetBase().cuda() for i in range(len(args.src))] 
    netB_list = [network.feat_bootleneck(type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))] 
    netC_list = [network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netDC_inter = network.domain_classifier(domain_num = len(args.src), bottleneck_dim=args.bottleneck).cuda()
    #netDC_inter_adv = network.domain_classifier_adv(domain_num = len(args.src), bottleneck_dim=args.bottleneck).cuda()


    param_group = []

    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        # for k, v in netC_list[i].named_parameters():
        #     param_group += [{'params':v, 'lr':args.lr * args.lr_decay3}]

    for k, v in netDC_inter.named_parameters():
        param_group += [{'params':v, 'lr':args.lr}]

    # for k, v in netDC_inter_adv.named_parameters():
    #     param_group += [{'params':v, 'lr':args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)


    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    args.w = torch.ones(len(args.src))
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()
        
        #print(tar_idx)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                #netC_list[i].eval()
                if iter_num == 0:
                    temp1, temp2, alpha = obtain_label_alpha(dset_loaders['target_'], netF_list[i], netB_list[i], netC_list[i], args, obtain_prior=True)
                    args.w[i] = alpha
                else:
                    temp1, temp2 = obtain_label_alpha(dset_loaders['target_'], netF_list[i], netB_list[i], netC_list[i], args, obtain_prior=False)
                temp1 = torch.from_numpy(temp1).cuda()#center points
                temp2 = torch.from_numpy(temp2).cuda()#features
                initc.append(temp1)#
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()
                #netC_list[i].train()
            if iter_num == 0:
                args.w = args.w/torch.sum(args.w)
                w = args.w
                print('Initialized weights:', args.w)
            else:
                args.w = w
        inputs_test = inputs_test.cuda()

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        src_domain_labels = torch.zeros(inputs_test.shape[0]*len(args.src), 1)

        start_each_domain = True
        for i in range(len(args.src)):
            features_test = netB_list[i](netF_list[i](inputs_test))
            outputs_test = netC_list[i](features_test)
            weights_numerator = netDC_inter(features_test)

            weights_test = weights_numerator
            softmax_weights = nn.Softmax(dim=1)(weights_test)

            domain_weight = softmax_weights[:, i]
            domain_weight = domain_weight.mean(dim=0)
            outputs_all[i] = outputs_test
            weights_all[:, i] = domain_weight*args.w[i]
            if start_each_domain:
                domain_preidctions = weights_test
                src_domain_labels = torch.full((inputs_test.shape[0], 1), int(i))
                start_each_domain = False
            else:
                src_domain_labels = torch.cat((src_domain_labels, torch.full((inputs_test.shape[0], 1), int(i))), 0)
                domain_preidctions = torch.cat((domain_preidctions, weights_test), 0)
        

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        outputs_all = torch.transpose(outputs_all, 0, 1)
        z_ =  weights_all[0:1][0]
        #print(z_)

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])
        #print(tar_idx.max())
        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            #print(temp[tar_idx, :].size())
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                #print(initc[i].size(),tar_idx.size())
                initc_ = initc_ + z_[i] * initc[i].float()
                #print(all_feas[i].size(),tar_idx.size())
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w.cuda(), pred.cuda())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.dc_loss:
            src_domain_labels = src_domain_labels.view(-1)
            domain_classification_loss = nn.CrossEntropyLoss(weight=args.w.cuda())(domain_preidctions.cuda(), src_domain_labels.cuda())
            #domain_classification_loss_adv = nn.CrossEntropyLoss()(domain_preidctions_adv.cuda(), src_domain_labels.cuda())

            #domain_classification_loss = nn.CrossEntropyLoss()(domain_preidctions.cuda(), src_domain_labels.cuda())
            total_domain_loss = args.dc_loss_par*domain_classification_loss #+ 0.4*domain_classification_loss_adv

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out.cuda()))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0).cuda()
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            #classifier_loss += im_loss
        


        total_loss = classifier_loss + total_domain_loss + im_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                netDC_inter.eval()
            acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netDC_inter, args)
            log_str = 'Iter:{}/{}; Classification Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
            print(log_str+'\n')

            log_loss = 'Iter:{}/{}; source loss = {:.4f}, im loss = {:.4f}, inter domain loss = {:.4f}'.format(iter_num, max_iter, classifier_loss, im_loss, domain_classification_loss)
            print(log_loss+'\n')

            args.out_file.write(log_str + '\n')
            args.out_file.write(log_loss + '\n')
            args.out_file.flush()
            #log_loss = 'Task: {}, Iter:{}/{}; source loss = {:.4f}, im loss = {:.4f}'.format(args.name, iter_num, max_iter, source_classifier_loss, im_loss)

            # for i in range(len(args.src)):
            #     torch.save(netF_list[i].state_dict(), osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
            #     torch.save(netB_list[i].state_dict(), osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"))
            #     torch.save(netC_list[i].state_dict(), osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
            #     torch.save(netG_list[i].state_dict(), osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"))

def obtain_label_alpha(loader, netF, netB, netC, args, obtain_prior=True):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
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
    if obtain_prior:
        alpha = utils.obtain_domain_prior(all_fea, all_output, args)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    #return pred_label.astype('int')
    if obtain_prior:
        return initc, all_fea, alpha
    else:
        return initc, all_fea


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netDC_inter, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)
            
            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                
                weights_numerator = netDC_inter(features)#Numerator and denominator
                #weights_denominator = netG_list(features)
                weights_test = weights_numerator#/weights_denominator
                softmax_weights = nn.Softmax(dim=1)(weights_test)
                domain_weight = softmax_weights[:, i]
                domain_weight = domain_weight.mean(dim=0)
                weights_all[:, i] = domain_weight*args.w[i]
                outputs_all[i] = outputs
                #weights_all[:, i] = domain_weight.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16
            #print(weights_all[0:1][0])
            weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
            #print(outputs_all.size())
            outputs_all = torch.transpose(outputs_all, 0, 1)
            #print(outputs_all.size())

            z_ =  weights_all[0:1][0]
            #print(z_)
            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    _, predict = torch.max(all_output, 1)
    print(z_)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ours')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='DigitFive', choices=['office31', 'office-home', 'office-caltech', 'DigitFive'])
    parser.add_argument('--lr', type=float, default=1*1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--pre_step', type=int, default=50, help="pretrain step of domain classifier")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--dc_loss', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--dc_loss_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)
    parser.add_argument('--ratio', type=float, default=0.5, help="the ratio of selected data")
    parser.add_argument('--K', type=int, default=20, help="the number of selected neighbors")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/adapt_ours')
    parser.add_argument('--output_src', type=str, default='ckps/source/uda')
    args = parser.parse_args()
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office31':
        names = ['amazon', 'dslr' , 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DigitFive':
        names = ['mnist_data', 'mnistm_with_label', 'svhn_train_32x32', 'synth_train_32x32', 'usps_28x28']
        args.class_num = 10

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = '/home/zhongyi/data/domain_adaptation/classification/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.mat'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.mat'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i].upper()))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.dset, names[args.t].upper())

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ctime = time.localtime()
    year, month, day, hour, miniute = ctime.tm_year, ctime.tm_mon, ctime.tm_mday, ctime.tm_hour, ctime.tm_min
    args.savename = 'par_' + str(args.cls_par) + '_' + str(year) + '-' + str(month) + '-' + str(day) + '-' + str(hour) + '-' + str(miniute)

    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_target(args)

