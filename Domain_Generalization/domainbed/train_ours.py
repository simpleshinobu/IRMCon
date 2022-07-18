# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
# os.chdir("../../")
import sys
sys.path.append("../")
import random
import sys
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL
import torch
import torchvision
import torch.optim as optim
import torch.utils.data
import math
from utils import GeneralizedCELoss, info_nce_loss, penalty_contra
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
#Note the default setting is for PACS envs1, the details of other settings are in the following.
#Please focus on the loss of training IRMCon, lr 0.0008 is better than 0.001 at some time if 0.001 cannot decrease the contrastive loss


def save_checkpoint(filename, model):
    save_dict = {
        "args": vars(args),
        "model_input_shape": dataset.input_shape,
        "model_num_classes": dataset.num_classes,
        "model_num_domains": len(dataset) - len(args.test_envs),
        "model_hparams": hparams,
        "model_dict": model.state_dict()
    }
    torch.save(save_dict, os.path.join(args.output_dir, filename))

def accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device),y.to(device)
            p = network.predict(x)
            batch_weights = torch.ones(len(x)).to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    return correct / total
def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, lr):
    p = (batch_id + 1 + epoch * total_batches) / (args.warm_epochs * total_batches)
    lr = p * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def accuracy_bias(model_contra, model_b, loader, device):
    correct = 0
    total = 0
    model_b.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device),y.to(device)
            features_temp = model_contra.network(x)
            features_temp = features_temp.detach()
            p = model_b(features_temp)
            batch_weights = torch.ones(len(x)).to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    model_b.train()
    return correct / total

def set_idx_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.idx_flag = value
    temp_set.datasets[1].underlying_dataset.idx_flag = value
    temp_set.datasets[2].underlying_dataset.idx_flag = value
def set_training_aug_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.training_aug = value
    temp_set.datasets[1].underlying_dataset.training_aug = value
    temp_set.datasets[2].underlying_dataset.training_aug = value
def set_training_dataset_idx_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.dataset_idx = value
    temp_set.datasets[1].underlying_dataset.dataset_idx = value
    temp_set.datasets[2].underlying_dataset.dataset_idx = value
def set_second_image_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.second_image = value
    temp_set.datasets[1].underlying_dataset.second_image = value
    temp_set.datasets[2].underlying_dataset.second_image = value
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    ###ours params
    parser.add_argument('--ERM_lr', type=float,default=0.001) #
    parser.add_argument('--weight_decay', type=float,default=5e-5)
    parser.add_argument('--batch_size', type=int,default=128)
    parser.add_argument('--workers', type=int,default=8)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--checkpoint_freq', type=int,default=1)
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0])
    #contrastive settings
    parser.add_argument('--contra_dim', type=int, default=64)
    parser.add_argument('--irm_weight', type=float, default=0.1)
    parser.add_argument('--batch_size_contra', type=int, default=480)
    parser.add_argument('--stage0_epochs', type=int, default=80)
    parser.add_argument('--stage1_epochs', type=int, default=150) #300 for PS

    parser.add_argument('--contra_lr', type=float, default=0.0008)
    parser.add_argument('--save_epoch', type=int, default=50)
    parser.add_argument('--print_frequency_contra', type=int, default=40)
    parser.add_argument('--lr_final_bias', type=float, default=0.005)
    parser.add_argument('--select_rear', type=int, default=10)
    parser.add_argument('--select_major', type=int, default=28) #control the GPU Mem under 16G

    parser.add_argument('--weight_base', type=float, default=0.5)
    parser.add_argument('--start_reweight_epoch', default=50, type=int)
    parser.add_argument('--classes_num', type=int, default=7)
    parser.add_argument('--n_start', type=int, default=3)
    parser.add_argument('--start_run', type=int, default=0)

    parser.add_argument('--schedule', type=str, default="cosine")
    parser.add_argument('--warm_epochs', type=int, default=0)
    parser.add_argument('--data_dir', type=str,default="./data/")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams_seed', type=int, default=0,help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,help='Trial number (used for seeding split_dataset and ''random_hparams).')
    parser.add_argument('--seed', type=int, default=0,help='Seed for everything else')

    parser.add_argument('--output_dir_first', type=str, default="log/")
    parser.add_argument('--add_note', type=str, default="pacs_ours")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[3]) ##0,1,2,3 for  ['art_painting', 'cartoon', 'photo', 'sketch']
    parser.add_argument('--no_pretrain', type=bool, default=True)
    parser.add_argument('--use_res18', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    start_step = 0
    algorithm_dict = None
    args.output_dir = os.path.join(args.output_dir_first, args.dataset + "_" + "test" + str(args.test_envs[0]) + "_" + args.add_note)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    final_test_accs1 = []
    final_test_accs2 = []
    if args.no_pretrain:
        hparams["pretrained"] = False
    if args.use_res18:
        hparams["resnet18"] = True
    for start in range(args.start_run, args.n_start):
        print("running {}/{}".format(start+1, args.n_start))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if args.dataset in vars(datasets):
            dataset = vars(datasets)[args.dataset](args.data_dir,
                args.test_envs, hparams)
        else:
            raise NotImplementedError

        in_splits = []
        out_splits = []
        uda_splits = []
        for env_i, env in enumerate(dataset):
            uda = []
            out, in_ = misc.split_dataset(env,int(len(env)*0.2),
                misc.seed_hash(args.trial_seed, env_i))

            if env_i in args.test_envs:
                uda, in_ = misc.split_dataset(in_,int(len(in_)*0),
                    misc.seed_hash(args.trial_seed, env_i))

            in_weights, out_weights, uda_weights = None, None, None

            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))

        #concatanate split
        from torch.utils.data import ConcatDataset
        train_set_list = []
        for i, (dataset_temp, _) in enumerate(in_splits):
            if i not in args.test_envs:
                train_set_list.append(dataset_temp)

        train_split_set = ConcatDataset(train_set_list)
        train_loader0 = torch.utils.data.DataLoader(
            train_split_set, batch_size=args.batch_size//2, shuffle=True, num_workers=args.workers)
        train_loader = torch.utils.data.DataLoader(
            train_split_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        train_loader_contra = torch.utils.data.DataLoader(
            train_split_set, batch_size=args.batch_size_contra, shuffle=True, num_workers=args.workers, drop_last=True)
        print("test env", args.test_envs)
        if args.dataset == "PACS":
            print("test env number 0,1,2,3 for ['art_painting', 'cartoon', 'photo', 'sketch']")

        #make eval_test
        eval_test_set = out_splits[args.test_envs[0]][0]
        eval_train_list = []
        for i, (dataset_temp, _) in enumerate(out_splits):
            if i not in args.test_envs:
                eval_train_list.append(dataset_temp)
        eval_train_set = ConcatDataset(eval_train_list)
        #make final test
        real_test_set = in_splits[args.test_envs[0]][0]

        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS)
            for env in [train_split_set, eval_train_set, eval_test_set, real_test_set]]

        eval_loader_names = ["0.train", "1.eval1","2.eval2","3.test"]
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        checkpoint_vals = collections.defaultdict(lambda: [])


        #training stage
        last_results_keys = None
        best_accs = {"best_eval1":0,"best_test1":0,"best_eval2":0,"best_test2":0,"best_iters1":0,"best_iters2":0}
        set_idx_flag(train_split_set, True)
        set_training_aug_flag(train_split_set, True)
        #training stage0
        print("create model for first stage to generate weights")
        model_ERM = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
        model_ERM.to(device)
        model_ERM.network = torch.nn.DataParallel(model_ERM.network, device_ids= args.gpu_list)
        ERM_optimizer0 = torch.optim.Adam(
            model_ERM.network.parameters(),
            lr=0.001, #default
            weight_decay=args.weight_decay
        )
        train_num = 0
        for epoch in range(args.stage0_epochs):
            for batch_idx, (x, y, idx) in enumerate(train_loader0):
                x, y = x.cuda(), y.cuda()
                loss = F.cross_entropy(model_ERM.network(x), y)
                ERM_optimizer0.zero_grad()
                loss.backward()
                ERM_optimizer0.step()

            if epoch % 10 == 0:
                print("epoch:", epoch, "loss:", loss.item())
        #get weights
        set_training_aug_flag(train_split_set, False)
        set_training_dataset_idx_flag(train_split_set, True)
        train_loader0 = torch.utils.data.DataLoader(
            train_split_set, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers)
        weights_all = torch.zeros(4,5000)
        print("generating weight")
        for batch_idx, (x, target, idx , datasets_idx) in enumerate(train_loader0):
            x, target = x.cuda(), target.cuda()
            output = model_ERM.network(x)
            logits_soft = output.detach().softmax(-1)
            mask_gt = torch.zeros_like(logits_soft)
            mask_gt[range(mask_gt.size(0)), target] = 1
            estimated_weight = ((1 - mask_gt) * logits_soft).sum(-1)
            for b in range(x.size(0)):
                weights_all[datasets_idx[b],idx[b]] = estimated_weight[b].detach().cpu()
        #start IRMCon
        set_second_image_flag(train_split_set, True)
        set_training_aug_flag(train_split_set, True)
        print("create model for second stage for contrastive")
        model_contra = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)

        model_contra.network[1] = nn.Linear(512, args.contra_dim)
        model_contra.to(device)
        model_contra.network = torch.nn.DataParallel(model_contra.network, device_ids= args.gpu_list)
        optimizer_contra = optim.Adam(model_contra.network.parameters(), lr=args.contra_lr)

        print("START contrastive")
        for epoch in range(args.stage1_epochs):
            for batch_idx, (datas, target, idx, datasets_idx) in enumerate(train_loader_contra):
                (imgs1, imgs2) = datas
                imgs1 = imgs1.cuda()
                imgs2 = imgs2.cuda()
                weight = torch.zeros_like(idx).float()
                for b in range(imgs1.size(0)):
                    weight[b] = weights_all[datasets_idx[b], idx[b]]
                weight = weight.cuda()
                idx_selected_all = []
                for k in range(args.classes_num):
                    list_temp = (target == k).nonzero(as_tuple=False).view(-1)
                    weight_temp = weight[list_temp]
                    sorted_weight, idxinweight = torch.sort(weight_temp, descending=True)
                    seleted_idx0 = idxinweight[args.select_rear:][
                        torch.randperm(idxinweight[args.select_rear:].size(0))[:args.select_major]]
                    seleted_idx = torch.cat((idxinweight[:args.select_rear], seleted_idx0), dim=0)
                    idx_real = list_temp[seleted_idx]
                    idx_selected_all.append(idx_real)
                train_nll = 0
                train_penalty = 0
                for k in range(args.classes_num):
                    temp_list = idx_selected_all[k]
                    if temp_list.size(0) == 0:
                        continue
                    temp_imgs1 = imgs1[temp_list]
                    temp_imgs2 = imgs2[temp_list]
                    features1 = model_contra.network(temp_imgs1)
                    features2 = model_contra.network(temp_imgs2)
                    logits, labels = info_nce_loss(torch.cat((features1, features2), dim=0), features1.size(0),
                                                   temperature=1)
                    train_nll += torch.nn.CrossEntropyLoss()(logits, labels)
                    train_penalty += penalty_contra(logits, labels, torch.nn.CrossEntropyLoss()).mean()

                train_nll = train_nll / float(args.classes_num)
                train_penalty = train_penalty / float(args.classes_num)
                loss = (train_nll + train_penalty * args.irm_weight)
                optimizer_contra.zero_grad()
                loss.backward()
                optimizer_contra.step()
            if (epoch % args.print_frequency_contra == 0 or epoch == args.stage1_epochs - 1):
                print("stage1, epoch:", epoch, "loss nll:", train_nll.item(),
                      "loss penalty:", train_penalty.item())
            if (epoch % args.save_epoch == 0 or epoch == args.stage1_epochs - 1) and epoch != 0:
                save_path = args.output_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = "starts_{}_model_contra_epoch{}.torch".format(start, epoch)
                save_name = os.path.join(save_path, save_name)
                torch.save({'state': model_contra.state_dict()}, save_name)
        set_second_image_flag(train_split_set, False)
        model_contra.eval()
        model_b = nn.Sequential(nn.Linear(args.contra_dim, args.contra_dim * 2),
                                nn.LeakyReLU(0.2),
                                nn.Linear(args.contra_dim * 2, 10)).cuda()
        model_b = torch.nn.DataParallel(model_b, device_ids=args.gpu_list).cuda()
        if args.no_pretrain:
            hparams["pretrained"] = False
        if args.use_res18:
            hparams["resnet18"] = True
        print("create final model for domain generation")
        model_ERM = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
        model_ERM.network = torch.nn.DataParallel(model_ERM.network, device_ids=args.gpu_list).cuda()

        optimizer_d = torch.optim.Adam(
            model_ERM.network.parameters(),
            lr=args.ERM_lr,  #default
            weight_decay=args.weight_decay
        )
        optimizer_b = optim.Adam(model_b.parameters(), lr=args.lr_final_bias)
        criterion_bias = GeneralizedCELoss()
        criterion_train = nn.CrossEntropyLoss(reduction="none")

        for epoch in range(args.epochs):
            set_idx_flag(train_split_set, True)
            set_training_aug_flag(train_split_set, True)
            set_training_dataset_idx_flag(train_split_set, False)
            for batch_idx, (images, target, idx) in enumerate(train_loader):
                images = images.cuda()
                target = target.cuda()

                features_temp = model_contra.network(images)
                features_temp = features_temp.detach()
                logit_b = model_b(features_temp)
                logit_d = model_ERM.network(images)
                loss_b = criterion_train(logit_b, target).detach()
                loss_d = criterion_train(logit_d, target).detach()
                loss_weight = loss_b / (loss_b + loss_d + 1e-8) + args.weight_base

                loss_b_update = criterion_bias(logit_b, target)
                if epoch < args.start_reweight_epoch:
                    loss_d_update = criterion_train(logit_d, target)
                else:
                    loss_weight = (loss_b / (loss_b + loss_d + 1e-8)) + args.weight_base
                    loss_d_update = criterion_train(logit_d, target) * loss_weight.cuda().detach()
                loss = loss_b_update.mean() + loss_d_update.mean()

                optimizer_b.zero_grad()
                optimizer_d.zero_grad()
                loss.backward()
                optimizer_b.step()
                optimizer_d.step()
                step_vals = {'loss': loss.item()}

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            if (epoch % args.checkpoint_freq == 0) or (epoch == args.epochs - 1):
                results = {'epoch': epoch,'lr': optimizer_d.param_groups[0]['lr'],}
                set_idx_flag(train_split_set, False)
                set_training_aug_flag(train_split_set, False)

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                evals = zip(eval_loader_names, eval_loaders)
                accu_temp_list = []
                for i, (name, loader) in enumerate(evals):
                    acc = accuracy(model_ERM, loader, device)
                    acc_bias = accuracy_bias(model_contra, model_b, loader, device)
                    results[name+'_acc'] = acc
                    results[name+'H_acc'] = acc_bias
                    accu_temp_list.append(acc)
                if accu_temp_list[1] > best_accs["best_eval1"]:
                    best_accs["best_eval1"] = accu_temp_list[1]
                    best_accs["best_test1"] = accu_temp_list[-1]
                    best_accs["best_iters1"] = epoch
                    save_checkpoint('model_select_from_train_{}.pkl'.format(start),model_ERM)
                if accu_temp_list[2] > best_accs["best_eval2"]:
                    best_accs["best_eval2"] = accu_temp_list[2]
                    best_accs["best_test2"] = accu_temp_list[-1]
                    best_accs["best_iters2"] = epoch
                    save_checkpoint('model_select_from_test_{}.pkl'.format(start),model_ERM)

                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                    colwidth=12)

                results.update({'hparams': hparams,'args': vars(args)})

                epochs_path = os.path.join(args.output_dir, 'results_{}.jsonl'.format(start))
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")
                checkpoint_vals = collections.defaultdict(lambda: [])
        print("best accu selected from train domain at {}: #### {} ####; best accu selected from test domain at {}: #### {} ####"
              .format(best_accs["best_iters1"],best_accs["best_test1"] ,best_accs["best_iters2"],best_accs["best_test2"] ))

        final_test_accs1.append(best_accs["best_test1"])
        final_test_accs2.append(best_accs["best_test2"])
        print("accu(test selected from train mean-std):{}-{}"
              .format(round(np.mean(final_test_accs1).item(), 4), round(np.std(final_test_accs1).item(), 4)))
        print("accu(test selected from test uniform mean-std):{}-{}"
              .format(round(np.mean(final_test_accs2).item(), 4),
                      round(np.std(final_test_accs2).item(), 4)))

