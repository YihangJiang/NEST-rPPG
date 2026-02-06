# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
import torch
import MyDataset
import MyLoss
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from datetime import datetime
import os
import time
from utils import Logger, time_to_str
from timeit import default_timer as timer
import time
import random

TARGET_DOMAIN = {'VIPL': ['V4V',  'PURE', 'BUAA', 'UBFC'], \
                 'V4V': ['VIPL',  'PURE', 'BUAA', 'UBFC'], \
                 'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC'], \
                 'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC'], \
                 'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap']}

if __name__ == '__main__':


    args = utils.get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Source: single domain if --src set, else all 4 except target
    if getattr(args, 'src', None) is not None:
        Source_domain_Names = [args.src]
        print('Single source domain:', args.src)
    else:
        Source_domain_Names = TARGET_DOMAIN[args.tgt]
    num_sources = len(Source_domain_Names)
    root_file = r'./STMap/'
    # Build source configs (list of dicts: name, fileRoot, saveRoot, map)
    source_configs = []
    for name in Source_domain_Names:
        fn = FILEA_NAME[name]
        source_configs.append({
            'name': name,
            'fileRoot': root_file + fn[0],
            'saveRoot': root_file + 'STMap_Index/' + fn[1],
            'map': fn[2] + '.png'
        })

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.png'
    
    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    src_suffix = ('_src' + args.src) if getattr(args, 'src', None) else ''
    rPPGNet_name = 'rPPGNet_' + Target_name + src_suffix + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + rPPGNet_name + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')

    # 数据集
    if args.reData == 1:
        Target_index = os.listdir(Target_fileRoot)
        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)
        for cfg in source_configs:
            idx_list = os.listdir(cfg['fileRoot'])
            MyDataset.getIndex(cfg['fileRoot'], idx_list, cfg['saveRoot'], cfg['map'], 10, frames_num)

    source_dbs = []
    for cfg in source_configs:
        db = MyDataset.Data_DG(root_dir=cfg['saveRoot'], dataName=cfg['name'], \
                              STMap=cfg['map'], frames_num=frames_num, args=args)
        print('Loaded source dataset:', cfg['name'], 'from', cfg['saveRoot'], 'num_samples =', len(db))
        source_dbs.append(db)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  STMap=Target_map, frames_num=frames_num, args=args)
    print('Loaded target dataset:', Target_name, 'from', Target_saveRoot, 'num_samples =', len(Target_db))

    src_loaders = [DataLoader(db, batch_size=batch_size_num, shuffle=True, num_workers=num_workers) for db in source_dbs]
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    BaseNet = model.BaseNet()

    if reTrain == 1:
        BaseNet = torch.load('./Result_Model/' + rPPGNet_name, map_location=device)
        print('load ' + rPPGNet_name + ' right')
    BaseNet.to(device=device)
    optimizer_rPPG = torch.optim.Adam(BaseNet.parameters(), lr=learning_rate)
    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
    loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
    loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
    loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)
    src_iters = [loader.__iter__() for loader in src_loaders]
    src_iter_per_epochs = [len(itr) for itr in src_iters]

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    # Training
    BaseNet.train()
    for iter_num in range(max_iter + 1):
        # Reset iterators at epoch boundaries
        for i in range(num_sources):
            if iter_num > 0 and (iter_num % src_iter_per_epochs[i] == 0):
                src_iters[i] = src_loaders[i].__iter__()

        ######### data prepare #########
        # Load data from all source domains
        src_data_list, src_bvp_list, src_HR_rel_list = [], [], []
        src_data_aug_list, src_bvp_aug_list, src_HR_rel_aug_list = [], [], []
        src_batch_sizes = []
        
        for i in range(num_sources):
            data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug = src_iters[i].__next__()
            src_batch_sizes.append(data.shape[0])
            src_data_list.append(Variable(data).float().to(device=device))
            src_bvp_list.append(Variable(bvp).float().to(device=device).unsqueeze(dim=1))
            src_HR_rel_list.append(Variable(torch.Tensor(HR_rel)).float().to(device=device))
            src_data_aug_list.append(Variable(data_aug).float().to(device=device))
            src_bvp_aug_list.append(Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1))
            src_HR_rel_aug_list.append(Variable(torch.Tensor(HR_rel_aug)).float().to(device=device))

        optimizer_rPPG.zero_grad()
        # Concatenate all source data
        input = torch.cat(src_data_list, dim=0)
        input_aug = torch.cat(src_data_aug_list, dim=0)
        bvp_pre, HR_pr, av = BaseNet(input)
        bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(input_aug)

        # Split predictions back by source domain
        bvp_pre_split = []
        HR_pr_split = []
        bvp_pre_aug_split = []
        HR_pr_aug_split = []
        start_idx = 0
        for bs in src_batch_sizes:
            end_idx = start_idx + bs
            bvp_pre_split.append(bvp_pre[start_idx:end_idx])
            HR_pr_split.append(HR_pr[start_idx:end_idx])
            bvp_pre_aug_split.append(bvp_pre_aug[start_idx:end_idx])
            HR_pr_aug_split.append(HR_pr_aug[start_idx:end_idx])
            start_idx = end_idx

        # Calculate losses for each source domain
        src_losses = []
        src_losses_aug = []
        for i in range(num_sources):
            src_loss_i = MyLoss.get_loss(bvp_pre_split[i], HR_pr_split[i], src_bvp_list[i], 
                                         src_HR_rel_list[i], source_configs[i]['name'],
                                         loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_aug_i = MyLoss.get_loss(bvp_pre_aug_split[i], HR_pr_aug_split[i], 
                                            src_bvp_aug_list[i], src_HR_rel_aug_list[i],
                                            source_configs[i]['name'], loss_func_NP, loss_func_L1, args, iter_num)
            src_losses.append(src_loss_i)
            src_losses_aug.append(src_loss_aug_i)

        # Supervised loss
        src_loss_sum = sum(src_losses)
        src_loss_aug_sum = sum(src_losses_aug)
        HR_rels = torch.cat(src_HR_rel_list, dim=0)
        HR_rel_augs = torch.cat(src_HR_rel_aug_list, dim=0)
        # NEST losses (structure / domain / temporal alignment)
        loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
        loss_DM = loss_func_NEST_DM(av, av_aug)
        loss_TA = loss_func_NEST_TA(torch.cat((av, av_aug), dim=0), torch.cat((HR_rels, HR_rel_augs), dim=0))

        # If you want to disable these three losses (purely supervised training),
        # keep them as tensors but zero them out so logging and .backward() remain valid.
        loss_CM = loss_CM * 0.0
        loss_DM = loss_DM * 0.0
        loss_TA = loss_TA * 0.0
        src_loss_aug_sum = src_loss_aug_sum * 0.0

        k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0

        loss = src_loss_sum + src_loss_aug_sum + 0.1 * k * loss_TA + 0.001 * k * loss_CM + 0.01 * k * loss_DM
        if torch.sum(torch.isnan(loss)) > 0:
            print('Nan')
            break
        else:
            loss.backward()
            optimizer_rPPG.step()
        if iter_num % 100 == 0:
            log_line = 'Train Inter:' + str(iter_num) + ' | loss:  ' + str(loss.data.cpu().numpy())
            for i in range(num_sources):
                log_line += ' |' + source_configs[i]['name'] + ' : ' + str(src_losses[i].data.cpu().numpy())
            log_line += ' |' + 'CM' + ' : ' + str(loss_CM.data.cpu().numpy()) \
                       + ' |' + 'DM' + ' : ' + str(loss_DM.data.cpu().numpy()) \
                       + ' |' + 'TA' + ' : ' + str(loss_TA.data.cpu().numpy()) \
                       + ' |' + time_to_str(timer() - start, 'min')
            log.write(log_line)
            log.write('\n')

    # Testing
    BaseNet.eval()
    HR_pr_temp = []
    HR_rel_temp = []
    BVP_ALL = []
    BVP_PR_ALL = []
    for step, (data, bvp, HR_rel, _, _, _) in enumerate(tgt_loader):
        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device)
        HR_rel = Variable(HR_rel).float().to(device=device)
        bvp = bvp.unsqueeze(dim=1)
        Wave = bvp
        Wave_pr, HR_pr, av = BaseNet(data)

        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
        BVP_ALL.extend(Wave.data.cpu().numpy())
        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())




    if not os.path.exists('./Result'):
        os.makedirs('./Result')
    io.savemat('./Result/' + rPPGNet_name + 'HR_pr.mat', {'HR_pr': HR_pr_temp})
    io.savemat('./Result/' + rPPGNet_name + 'HR_rel.mat', {'HR_rel': HR_rel_temp})
    io.savemat('./Result/' + rPPGNet_name + 'WAVE_ALL.mat',
               {'Wave': BVP_ALL})
    io.savemat('./Result/' + rPPGNet_name + 'WAVE_PR_ALL.mat',
               {'Wave': BVP_PR_ALL})
    if not os.path.exists('./Result_Model'):
        os.makedirs('./Result_Model')
    model_path = './Result_Model/' + rPPGNet_name
    torch.save(BaseNet, model_path)
    print('Saved model:', os.path.abspath(model_path))
