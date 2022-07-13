import torchvision.transforms as transforms
import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch import nn, optim
from dataset.dataset import our_dataset_bar, split_test
from models.backbones import ResNet18
from utils.utils import pretty_print, penalty_contra, info_nce_loss, GeneralizedCELoss, cosine_schedule
import warnings
warnings.filterwarnings("ignore")
##base settings
parser = argparse.ArgumentParser(description='BAR')
parser.add_argument('--contra_dim', type=int, default=64)
parser.add_argument('--irm_weight', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dim_final_multiple', type=int, default=2)
parser.add_argument('--lr_final_bias', type=float, default=0.001)
parser.add_argument('--test_interval', type=int, default=5)
parser.add_argument('--train_print_interval', type=int, default=5)
parser.add_argument('--print_frequency_contra', type=int, default=50)
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--batch_size_contra', type=int, default=330)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--warm_epochs', type=int, default=10)
parser.add_argument('--add_note', type=str, default="bar_ours")
parser.add_argument('--save_dir', type=str, default="log")
parser.add_argument('--schedule', type=str, default="cosine")
parser.add_argument('--ratio', type=float, default=0.05)
##ours settings
parser.add_argument('--weight_base', type=float, default=0.08) ##0.02 for 99% and 0.08 for 95%
parser.add_argument('--start_bias_reweight', type=int, default=60)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--eval_epoch', type=int, default=200)
parser.add_argument('--decay_epoch', type=int, default=160)
parser.add_argument('--decay_final_bias', type=int, default=60)

parser.add_argument('--stage0_epochs', type=int, default=100)
parser.add_argument('--stage0_decay', type=int, default=80)
parser.add_argument('--stage1_epochs', type=int, default=400)
###contrastive setting
parser.add_argument('--contra_lr', type=float, default=0.002)
parser.add_argument('--save_epoch', type=int, default=40)
parser.add_argument('--class_num', type=int, default=6)
parser.add_argument('--select_real', type=int, default=20)
parser.add_argument('--select_major', type=int, default=5)
parser.add_argument('--max_contra_num', type=int, default=30)
parser.add_argument('--workers', type=int, default=16)
args, unknown = parser.parse_known_args()
gpu_list = [0, 1]
def target_transform(target):
    return int(target)
def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, lr):
    p = (batch_id + 1 + epoch * total_batches) / (args.warm_epochs * total_batches)
    lr = p * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
from randaugment import RandAugment
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
transform_contra = transforms.Compose([
    transforms.RandomResizedCrop(args.image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(),
    transforms.ToTensor(),
    normalize
])
train_transform = transform_contra
test_transform = transforms.Compose([
    transforms.Resize(int(args.image_size*1.1)),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    normalize
])
final_train_accs = []
final_test_accs_uniform = []
final_val_accs_uniform = []
def evaluate(model, loader,flag="", use_feature=False, model_contra=None):
    right_num = 0
    total_num = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        if use_feature:
            output = model_contra.module.get_features_contra(data).detach()
            output = model(output)
        else:
            output = model(data)
        y_pred = output.softmax(-1).argmax(-1)
        right_num += (y_pred == target).sum().cpu().item()
        total_num += data.size(0)
    print(flag,"accu:",right_num/ total_num)
    return right_num/ total_num

for restart in range(args.n_restarts):
    split_test(args)
    train_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio),
                                set_name="train", transform=train_transform, save_flag=True)
    valid_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio),
                               set_name="valid", transform=test_transform, save_flag=True)
    test_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio),
                               set_name="test", transform=test_transform, save_flag=True)

    train_set_contra = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="train",
                                       transform=transform_contra, save_flag=True, second_image=True)
    train_set_clear = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="train",
                                      transform=test_transform, save_flag=True)

    train_loader_contra = torch.utils.data.DataLoader(
        train_set_contra, batch_size=args.batch_size_contra, shuffle=True, num_workers=args.workers * 2, drop_last=True)
    train_loader_clear = torch.utils.data.DataLoader(
        train_set_clear, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    acc_test_best = 0
    args.warmup_from = 0.
    args.warmup_to = args.lr
    torch.cuda.empty_cache()
    print("Restart index:", restart)
    criterion_stage0 = nn.CrossEntropyLoss()
    ##train a model for sample selection
    model_stage0 = ResNet18(args).cuda()
    model_stage0 = torch.nn.DataParallel(model_stage0, device_ids=gpu_list)
    optimizer = optim.Adam(model_stage0.parameters(), lr=0.002)
    train_set.idx_flag = True
    for epoch in range(args.stage0_epochs):
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            if epoch == args.stage0_decay:
                optimizer.param_groups[0]['lr'] *= 0.2
            data, target = data.cuda(), target.cuda()
            output = model_stage0(data)
            loss = criterion_stage0(output, target)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 30 == 0 and epoch != 0 or epoch == args.stage0_epochs - 1:
            print("stage0, train a model for sample selection, epoch:", epoch, "loss:", loss.item())
        ##test train accu
    acc_train = evaluate(model_stage0, train_loader_clear, flag="train")
    print("stage0 train accu{}".format(acc_train))
    train_set.idx_flag = False
    estimated_weight_all = None
    train_set_clear.idx_flag = True
    for batch_idx, (data, target, idx) in enumerate(train_loader_clear):
        model_stage0.eval()
        data, target = data.cuda(), target.cuda()
        output = model_stage0(data)
        logits_soft = output.detach().softmax(-1)
        mask_gt = torch.zeros_like(logits_soft)
        mask_gt[range(mask_gt.size(0)), target] = 1
        estimated_weight = ((1 - mask_gt) * logits_soft).sum(-1)
        if estimated_weight_all is None:
            estimated_weight_all = estimated_weight
        else:
            estimated_weight_all = torch.cat((estimated_weight_all, estimated_weight),dim=0)
    train_set_clear.idx_flag = False
    ##stage1 contrastive learning for IRMCon
    model_contra = ResNet18(args).cuda()
    model_contra = torch.nn.DataParallel(model_contra, device_ids=gpu_list)
    optimizer_contra = optim.Adam(model_contra.parameters(), lr=args.contra_lr)
    train_set_contra.weight_flag = True
    for epoch in range(args.stage1_epochs):
        ##sample from here, ramdom eliminate low weight samples
        sorted_weight, sorted_idx = torch.sort(estimated_weight_all,descending=True)
        total_num = estimated_weight_all.size(0)
        select_num = total_num//4
        selected_idx = sorted_idx[:select_num]
        remain_idx = sorted_idx[select_num:]
        selected_idx2 = remain_idx[torch.randperm(remain_idx.size(0))][:select_num]
        all_selected = torch.cat((selected_idx, selected_idx2), dim = 0)
        ##set all info of the dataset
        train_set_contra.weight_list = estimated_weight_all[all_selected].cpu().tolist()
        train_set_contra.image_list = [train_set_clear.image_list[k] for k in all_selected.tolist()]
        train_set_contra.label_list = [train_set_clear.label_list[k] for k in all_selected.tolist()]
        train_loader_contra = torch.utils.data.DataLoader(
            train_set_contra, batch_size=args.batch_size_contra, shuffle=True, num_workers=16,
            drop_last=True)
        for batch_idx, (datas, target, weight) in enumerate(train_loader_contra):
            (imgs1, imgs2) = datas
            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
            target = target.cuda()

            idx_selected_all = []
            idx_selected_all_list = []
            for k in range(args.class_num):
                list_temp = (target == k).nonzero().view(-1)
                weight_temp = weight[list_temp]
                sorted_weight, idxinweight = torch.sort(weight_temp, descending=True)
                seleted_idx0 = idxinweight[args.select_real:][torch.randperm(idxinweight[args.select_real:].size(0))[:args.select_major]]
                seleted_idx = torch.cat((idxinweight[:args.select_real],seleted_idx0),dim=0)
                idx_real = list_temp[seleted_idx]
                idx_selected_all.append(idx_real)
                idx_selected_all_list.extend(idx_real.tolist())
            ##add cross to the last
            len_all_list = len(idx_selected_all_list)
            select_final_idx = torch.Tensor(idx_selected_all_list).long()[
                torch.randperm(len_all_list)[:args.max_contra_num]].cuda()
            idx_selected_all.append(select_final_idx)
            train_nll = 0
            train_penalty = 0
            for k in range(args.class_num):
                temp_list = idx_selected_all[k]
                ##original:
                temp_imgs1 = imgs1[temp_list]
                temp_imgs2 = imgs2[temp_list]
                features1 = model_contra(temp_imgs1,need_contra_features=True)
                features2 = model_contra(temp_imgs2,need_contra_features=True)
                logits, labels = info_nce_loss(torch.cat((features1, features2), dim=0), features1.size(0),
                                               temperature=1)
                train_nll += torch.nn.CrossEntropyLoss()(logits, labels)
                train_penalty += penalty_contra(logits, labels, torch.nn.CrossEntropyLoss()).mean()
            train_nll = train_nll / float(args.class_num)
            train_penalty = train_penalty / float(args.class_num)
            loss = (train_nll + train_penalty * args.irm_weight)
            optimizer_contra.zero_grad()
            loss.backward()
            optimizer_contra.step()
        if (epoch % args.print_frequency_contra == 0 or epoch == args.stage1_epochs - 1) and epoch != 0:
            print("stage1 train IRMCon, epoch:", epoch, "loss nll:", train_nll.item(),
                  "loss penalty:", train_penalty.item())

        if (epoch%args.save_epoch == 0 or epoch == args.stage1_epochs -1):
            save_subdir = "{}_epochs_{}".format(args.add_note,args.epochs)
            save_path = os.path.join(args.save_dir, save_subdir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = "starts_{}_final_model_ccontra_epoch{}.torch".format(restart,epoch)
            save_name = os.path.join(save_path, save_name)
            torch.save({'state': model_contra.state_dict()}, save_name)
            ##stage1 over
    model_contra.eval()
    ##final stage train IPW
    model_b = nn.Sequential(nn.Linear(args.contra_dim, args.contra_dim * args.dim_final_multiple), nn.LeakyReLU(0.2),
                          nn.Linear(args.contra_dim * args.dim_final_multiple, 10)).cuda()
    model_b = torch.nn.DataParallel(model_b, device_ids=gpu_list)
    optimizer_b = optim.Adam(model_b.parameters(), lr=args.lr_final_bias)
    model_d = ResNet18(args).cuda()
    model_d = torch.nn.DataParallel(model_d, device_ids=gpu_list)
    optimizer_d  = optim.Adam(model_d.parameters(), lr=args.lr)


    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss()
    train_set.idx_flag = True
    best_acc = 0
    best_valid_acc = 0
    for epoch in range(args.epochs):
        if epoch == args.decay_final_bias or epoch == args.decay_final_bias * 2:
            optimizer_b.param_groups[0]['lr'] *= 0.2
        if epoch > args.warm_epochs and args.schedule == "cosine":
            cosine_schedule(optimizer_d, epoch, args, args.lr)
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            if epoch < args.warm_epochs:
                warmup_learning_rate(optimizer_b, epoch, batch_idx, len(train_loader), args.lr_final_bias)
                warmup_learning_rate(optimizer_d, epoch, batch_idx, len(train_loader), args.lr)
            data, target = data.cuda(), target.cuda()
            features_temp = model_contra(data, need_contra_features=True).detach()
            logit_b = model_b(features_temp)
            logit_d = model_d(data)
            loss_b = criterion(logit_b, target).detach()
            loss_d = criterion(logit_d, target).detach()
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            loss_b_update = bias_criterion(logit_b, target)
            if epoch < args.start_bias_reweight:
                loss_d_update = criterion(logit_d, target)
            else:
                loss_d_update = criterion(logit_d, target) * (loss_weight.cuda().detach() + args.weight_base)
            loss = loss_b_update.mean() + loss_d_update.mean()

            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()
        print("epoch",epoch,"loss",loss,"lr_b",optimizer_b.param_groups[0]['lr']
              ,"lr_d",optimizer_d.param_groups[0]['lr'])

        if epoch % args.test_interval == 0 and epoch !=0 or epoch == args.epochs - 1:
            train_set.idx_flag = False
            print("epochs:", epoch, "biased model:")
            model_b.eval()
            # evaluate for bias model
            # acc_train_b = evaluate(model_b, train_loader_clear, flag="train", use_feature=True, model_contra=model_contra.eval())
            # acc_test_b = evaluate(model_b, test_loader, flag="test" , use_feature=True, model_contra=model_contra.eval())
            model_b.train()
            print("epochs:", epoch, "unbiased model:")
            model_d.eval()
            train_acc = evaluate(model_d, train_loader_clear, flag="train")
            test_acc_valid = evaluate(model_d, valid_loader, flag="valid")
            test_acc_test = evaluate(model_d, test_loader, flag="test")
            model_d.train()
            if epoch > args.eval_epoch:
                if test_acc_valid >= best_valid_acc:
                    best_acc = test_acc_test
                    best_valid_acc = test_acc_valid
                    save_subdir = "{}".format(args.add_note)
                    save_path = os.path.join(args.save_dir, save_subdir)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_name = "starts_{}_final_model_d_best.torch".format(restart)
                    save_name = os.path.join(save_path, save_name)
                    torch.save({'state': model_d.state_dict()}, save_name)
                    print("best acc{}:".format(test_acc_test), "save model into {}".format(save_name))
            train_set.idx_flag = True
    final_train_accs.append(train_acc)
    final_test_accs_uniform.append(best_acc)
    final_val_accs_uniform.append(best_valid_acc)
    print("train:", round(np.mean(final_train_accs), 4), "test_uniform:", round(np.mean(final_test_accs_uniform), 4))
    print("accu(train mean-std):{}-{}"
          .format(round(np.mean(final_train_accs).item(), 4), round(np.std(final_train_accs).item(), 4)))
    print("accu(valid (best selected by valid) uniform mean-std):{}-{}"
          .format(round(np.mean(final_val_accs_uniform).item(), 4), round(np.std(final_val_accs_uniform).item(), 4)))
    print("accu(test uniform mean-std):{}-{}"
          .format(round(np.mean(final_test_accs_uniform).item(), 4), round(np.std(final_test_accs_uniform).item(), 4)))
    ##save models
    save_subdir = "{}".format(args.add_note)
    save_path = os.path.join(args.save_dir, save_subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name1 = "starts_{}_final_model_b.torch".format(restart)
    save_name1 = os.path.join(save_path, save_name1)
    torch.save({'state': model_b.state_dict()}, save_name1)
    save_name2 = "starts_{}_final_model_d.torch".format(restart)
    save_name2 = os.path.join(save_path, save_name2)
    torch.save({'state': model_d.state_dict()}, save_name2)
    save_name = "starts_{}_final_model_ccontra.torch".format(restart)
    save_name = os.path.join(save_path, save_name)
    torch.save({'state': model_contra.state_dict()}, save_name)



