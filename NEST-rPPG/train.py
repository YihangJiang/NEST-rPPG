# %%
# -*- coding: UTF-8 -*-
# Run as script: python train.py
# Run in Jupyter: execute cells in order (Cell 1 = config, Cell 2 = rest or run all)
%reload_ext autoreload
%autoreload 2
import torch
import numpy as np
import scipy.io as io
import MyDataset
import MyLoss
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from datetime import datetime
import os
import time
from utils.core import Logger, time_to_str, get_args
from utils import train_utils
from timeit import default_timer as timer
import time
import random
from types import SimpleNamespace

import config
# %%

# ============ Cell 1: Config (constants) – edit and run in Jupyter ============
# True = use config.py. False = use command-line args (python train.py ...).
_USE_JUPYTER_CONFIG = True

if _USE_JUPYTER_CONFIG:
    args = SimpleNamespace(
        GPU='0',
        num_workers=2,
        epochs=64 * 64 * 64,
        batchsize=100,
        lr=0.001,
        fold_num=5,
        fold_index=0,
        reTrain=0,
        max_iter=1000,
        seed=config.SEED,
        k1=1.0, k2=0.1, k3=1.0, k4=0.1, k5=1.0, k6=0.1, k7=0.1, k8=0.1,
        temporal_aug_rate=config.TEMPORAL_AUG_RATE,
        spatial_aug_rate=config.SPATIAL_AUG_RATE,
        form='Resize',
        weight=36,
        height=36,
        frames_num=256,
        tgt=config.TGT_DOMAIN,
        src=config.SRC_DOMAIN,
        loss_type=config.LOSS_TYPE,
        wave_sort_root=config.WAVE_SORT_ROOT,
    )
else:
    args = get_args()
    for attr, default in [
        ('loss_type', config.LOSS_TYPE),
        ('tgt', config.TGT_DOMAIN),
        ('src', config.SRC_DOMAIN),
        ('temporal_aug_rate', config.TEMPORAL_AUG_RATE),
        ('spatial_aug_rate', config.SPATIAL_AUG_RATE),
        ('wave_sort_root', config.WAVE_SORT_ROOT),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)
# ============ End Cell 1 ============

# ============ Cell 2: Training (run after config) ============
print("=" * 60)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Source: single domain if --src set, else all 4 except target
if getattr(args, 'src', None) is not None:
    Source_domain_Names = [args.src]
    print('Single source domain:', args.src)
else:
    Source_domain_Names = config.TARGET_DOMAIN[args.tgt]
num_sources = len(Source_domain_Names)
root_file = config.STMAP_PARENT_ROOT
index_base = config.STMAP_INDEX_BASE
# Build source configs (list of dicts: name, fileRoot, saveRoot, map)
source_configs = []
for name in Source_domain_Names:
    fn = config.FILEA_NAME[name]
    source_configs.append({
        'name': name,
        'fileRoot': os.path.join(root_file, fn[0]),
        'saveRoot': os.path.join(index_base, fn[1]),
        'map': fn[2] + '.png'
    })

FILE_Name = config.FILEA_NAME[args.tgt]
Target_name = args.tgt
Target_fileRoot = os.path.join(root_file, FILE_Name[0])
Target_saveRoot = os.path.join(index_base, FILE_Name[1])
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

os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
rPPGNet_name = config.build_run_name(
    tgt=Target_name,
    src=getattr(args, 'src', None),
    spatial=getattr(args, 'spatial_aug_rate', None),
    temporal=getattr(args, 'temporal_aug_rate', None),
    loss_type=getattr(args, 'loss_type', None),
)
log = Logger()
log.open(os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt'), mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

# 运行媒介
if torch.cuda.is_available():
    device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')
    print('on GPU')
else:
    device = torch.device('cpu')
    print('on CPU')
# %%
# 数据集 — always regenerate index from STMap folders
print("\n--- Dataset & index setup ---")
print("Regenerating indexes (clearing old index files, then building from STMap folders).")
for cfg in source_configs + [{'name': Target_name, 'saveRoot': Target_saveRoot}]:
    save_root = cfg['saveRoot']
    if os.path.exists(save_root):
        removed = 0
        for fname in os.listdir(save_root):
            fpath = os.path.join(save_root, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                removed += 1
        if removed:
            print("  Cleared %d index files in %s" % (removed, save_root))

def _subject_list(root):
    return sorted([f for f in os.listdir(root) if not f.startswith('.') and os.path.isdir(os.path.join(root, f))])
print("  Building target index: %s from %s" % (Target_name, Target_fileRoot))
Target_index = _subject_list(Target_fileRoot)
Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index,
                                   Target_saveRoot, Target_map, 10, frames_num)
for cfg in source_configs:
    print("  Building source index: %s from %s" % (cfg['name'], cfg['fileRoot']))
    idx_list = _subject_list(cfg['fileRoot'])
    MyDataset.getIndex(cfg['fileRoot'], idx_list, cfg['saveRoot'], cfg['map'], 10, frames_num)

print("Loading datasets (Data_DG) from index dirs...")
source_dbs = []
for cfg in source_configs:
    db = MyDataset.Data_DG(root_dir=cfg['saveRoot'], dataName=cfg['name'], \
                          STMap=cfg['map'], frames_num=frames_num, args=args)
    print('  Loaded source dataset:', cfg['name'], 'from', cfg['saveRoot'], 'num_samples =', len(db))
    source_dbs.append(db)
Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                              STMap=Target_map, frames_num=frames_num, args=args)
print('  Loaded target dataset:', Target_name, 'from', Target_saveRoot, 'num_samples =', len(Target_db))

print("Creating DataLoaders (batch_size=%d, num_workers=%d)..." % (batch_size_num, num_workers))
src_loaders = [DataLoader(db, batch_size=batch_size_num, shuffle=True, num_workers=num_workers) for db in source_dbs]
tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)
print("--- Done.\n")

# Save example figures: one sample per domain (STMap + BVP)
train_utils.save_example_figures(
    root_file,
    tgt_loader,
    src_loaders,
    source_configs,
    Target_name,
    tgt_db=Target_db,
    source_dbs=source_dbs,
    target_map=Target_map,
    example_fig_dir=os.path.join(config.RESULT_LOG_DIR, 'example_fig'),
)

# %%
BaseNet = model.BaseNet()

if reTrain == 1:
    BaseNet = torch.load(os.path.join(config.MODEL_DIR, rPPGNet_name), map_location=device)
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

# ----- Print info before training -----
print("\n" + "=" * 60)
print("TRAINING CONFIG")
print("=" * 60)
print("  Target domain:    ", Target_name)
print("  Target index:     ", Target_saveRoot)
print("  Target STMap:     ", Target_map)
print("  Source domains:   ", Source_domain_Names)
for i, cfg in enumerate(source_configs):
    print("    [%d] %s -> %s" % (i, cfg['name'], cfg['saveRoot']))
print("  Batch size:       ", batch_size_num)
print("  Max iterations:   ", max_iter)
print("  Frames per clip:  ", frames_num)
print("  Loss type:        ", getattr(args, 'loss_type', config.LOSS_TYPE))
print("  Device:           ", device)
print("  Model:            ", BaseNet.__class__.__name__)
total_src_samples = sum(len(db) for db in source_dbs)
print("  Source samples:   ", total_src_samples, " (target: %d)" % len(Target_db))
print("  Log file:         ", os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt'))
print("=" * 60 + "\n")

# Training
BaseNet.train()
for iter_num in range(max_iter + 1):
    # Reset iterators at epoch boundaries
    for i in range(num_sources):
        if iter_num > 0 and (iter_num % src_iter_per_epochs[i] == 0):
            src_iters[i] = src_loaders[i].__iter__()

    ######### data prepare #########
    src_data_list, src_bvp_list, src_HR_rel_list = [], [], []
    src_data_aug_list, src_bvp_aug_list, src_HR_rel_aug_list = [], [], []
    src_batch_sizes = []

    for i in range(num_sources):
        data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, _ = src_iters[i].__next__()
        src_batch_sizes.append(data.shape[0])
        src_data_list.append(Variable(data).float().to(device=device))
        src_bvp_list.append(Variable(bvp).float().to(device=device).unsqueeze(dim=1))
        src_HR_rel_list.append(Variable(torch.Tensor(HR_rel)).float().to(device=device))
        src_data_aug_list.append(Variable(data_aug).float().to(device=device))
        src_bvp_aug_list.append(Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1))
        src_HR_rel_aug_list.append(Variable(torch.Tensor(HR_rel_aug)).float().to(device=device))

    optimizer_rPPG.zero_grad()
    input = torch.cat(src_data_list, dim=0)
    input_aug = torch.cat(src_data_aug_list, dim=0)
    bvp_pre, HR_pr, av = BaseNet(input)
    bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(input_aug)

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

    src_loss_sum = sum(src_losses)
    src_loss_aug_sum = sum(src_losses_aug)
    HR_rels = torch.cat(src_HR_rel_list, dim=0)
    HR_rel_augs = torch.cat(src_HR_rel_aug_list, dim=0)
    loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
    loss_DM = loss_func_NEST_DM(av, av_aug)
    loss_TA = loss_func_NEST_TA(torch.cat((av, av_aug), dim=0), torch.cat((HR_rels, HR_rel_augs), dim=0))

    loss_type = getattr(args, 'loss_type', config.LOSS_TYPE)
    if loss_type == 'One':
        loss = src_loss_sum
    elif loss_type == 'TA':
        loss = src_loss_sum + loss_TA
    elif loss_type == 'CM':
        loss = src_loss_sum + loss_CM
    elif loss_type == 'DM':
        loss = src_loss_sum + loss_DM
    elif loss_type == 'All':
        loss = src_loss_sum + loss_TA + loss_CM + loss_DM
    else:
        loss = src_loss_sum + loss_TA
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
# %%
# Testing
BaseNet.eval()
HR_pr_temp = []
HR_rel_temp = []
BVP_ALL = []
BVP_PR_ALL = []
for step, (data, bvp, HR_rel, _, _, _, _) in enumerate(tgt_loader):
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

os.makedirs(config.RESULT_DIR, exist_ok=True)
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + 'HR_pr.mat'), {'HR_pr': HR_pr_temp})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + 'HR_rel.mat'), {'HR_rel': HR_rel_temp})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + 'WAVE_ALL.mat'), {'Wave': BVP_ALL})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + 'WAVE_PR_ALL.mat'), {'Wave': BVP_PR_ALL})
os.makedirs(config.MODEL_DIR, exist_ok=True)
model_path = os.path.join(config.MODEL_DIR, rPPGNet_name)
torch.save(BaseNet, model_path)
print('Saved model:', os.path.abspath(model_path))

try:
    wave_sort_out = os.path.join(config.WAVE_SORT_ROOT, Target_name, rPPGNet_name)
    train_utils.wave_sort_from_index(
        Target_saveRoot,
        np.array(BVP_ALL),
        np.array(BVP_PR_ALL),
        wave_sort_out,
    )
except Exception as e:
    print('Warning: Wave_sort failed:', repr(e))

# %%
