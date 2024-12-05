import sys
sys.path.insert(0, '.')
import time
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from model.fastforiens import EiffVit_seg
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.dataloader import getDataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import torch.nn.functional as F

# training parameter
lr_start = 1e-3
weight_decay = 0.025
max_iter = 90000
use_fp16 = True
warmup_iters = 1000
batchSize = 64

# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = (neg_num * 1.0 / sum_num).to(dtype=torch.float16)
    weight[neg_index] = (pos_num * 1.0 / sum_num).to(dtype=torch.float16)

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss

        return loss


def set_model(lb_ignore=255):
    net = EiffVit_seg(num_classes=2)
    net.cuda()
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr_start,
        betas=[0.9, 0.999],
        weight_decay=weight_decay,
    )
    return optim


def train():

    dl_train, dl_val = getDataLoader(batchSize)
    print(len(dl_train), len(dl_val))

    # model
    net= set_model()

    # optimizer
    optim = set_optimizer(net)

    # mixed precision training
    scaler = amp.GradScaler()

    criteria_pre = nn.CrossEntropyLoss()
    criteria_pre_1 = nn.L1Loss(reduction='none')
    criteria_pre_2 = nn.L1Loss()
    bd_criteria = BondaryLoss()

    # lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=max_iter, warmup_iter=warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    it = 0
    train_loader = iter(dl_train)

    plt_val_loss = []
    plt_train_loss = []
    plt_val_loss_area = []
    plt_train_loss_area = []

    train_loss = 0
    train_loss_area = 0

    AUC_val = []
    F1_val = []

    maxF1 = 0
    maxAUC = 0
    tic = time.time()
    net.train()
    while it < max_iter:
        try:
            im, lb, offset_gts, bd_gts = next(train_loader)
        except StopIteration:
            train_loader = iter(dl_train)
            continue
        im = im.cuda()
        lb = lb.cuda()
        bd_gts = (bd_gts.cuda()).to(dtype=torch.float16)
        offset_gts = (offset_gts.cuda()).to(dtype=torch.float16)

        optim.zero_grad()
        with amp.autocast(enabled=use_fp16):
            logits, bd_prd, offset_prd = net(im)
            loss_1 = criteria_pre(logits, lb.long())
            loss_4 = torch.mean(criteria_pre_1(offset_prd, offset_gts) * (lb.unsqueeze(dim=1)))
            loss_2 = bd_criteria(bd_prd, bd_gts.unsqueeze(dim=1))
        # 保存网络loss和loss_pre
        loss = loss_1 + loss_2 + 10 * loss_4
        train_loss += loss.item()
        train_loss_area += (loss_2.item() + 10 * loss_4.item())

        # 更新网络参数
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        if (it + 1) % 1000 == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0
                val_loss_area = 0
                val_f1 = 0
                val_auc = 0
                acc = 0

                for i ,data in enumerate(dl_val):
                    im, lb, offset_gts, bd_gts = data
                    im = im.cuda()
                    lb = lb.cuda()
                    bd_gts = (bd_gts.cuda()).to(dtype=torch.float16)
                    offset_gts = (offset_gts.cuda()).to(dtype=torch.float16)
                    with amp.autocast(enabled=use_fp16):
                        logits, bd_prd, offset_prd = net(im)
                        loss_1 = criteria_pre(logits, lb.long())
                        loss_4 = torch.mean(criteria_pre_1(offset_prd, offset_gts) * (lb.unsqueeze(dim=1)))
                        loss_2 = bd_criteria(bd_prd, bd_gts.unsqueeze(dim=1))
                        # 保存网络loss和loss_pre
                    loss = loss_1 + loss_2 + 10 * loss_4
                    val_loss += loss.item()
                    val_loss_area += (loss_2.item() + 10 * loss_4.item())

                    # 计算Val F1和AUC
                    logit = logits[0].unsqueeze(dim=0)
                    logit = torch.softmax(logit, dim=1)[:, 1, :, :]
                    result_mask = np.where(logit.cpu().detach().numpy() > 0.5, 1, 0).flatten()
                    gt_mask = lb[0].cpu().detach().numpy().flatten()
                    acc += (result_mask == gt_mask).sum() / result_mask.size
                    val_f1 += f1_score(gt_mask, result_mask)
                    result_mask = logit.cpu().detach().numpy().flatten()
                    gt_mask = lb[0].cpu().detach().numpy().flatten()
                    val_auc += roc_auc_score(gt_mask, result_mask)

                # Val阶段 数据整理打印

                plt_train_loss.append(train_loss / 1000)
                plt_train_loss_area.append(train_loss_area / 1000)
                AUC_val.append(val_auc / (len(dl_val)))
                F1_val.append(val_f1 / (len(dl_val)))
                acc = acc / (len(dl_val))
                plt_val_loss.append(val_loss / dl_val.__len__())
                plt_val_loss_area.append(val_loss_area / dl_val.__len__())

                print('[%03d/%03d] %2.2f sec(s) Loss: %3.6f Loss_area: %3.6f| Val AUC: %3.6f F1: %3.6f loss: %3.6f loss_area: %3.6f max_AUC: %3.6f max_F1: %3.6f ACC:%3.6f' % \
                    (it + 1, max_iter, time.time()-tic, plt_train_loss[-1], plt_train_loss_area[-1], AUC_val[-1], F1_val[-1], plt_val_loss[-1], plt_val_loss_area[-1], maxAUC,
                     maxF1, acc))

                tic = time.time()

                train_loss = 0
                train_loss_area = 0

                net.train()

        it += 1
        lr_schdr.step()



def main():
    train()


if __name__ == "__main__":
    main()
