
import argparse
import numpy as np
import torch
import os
from torch import nn, optim
from models.backbones import ResNet18
from models.model_ours import CNN_cifar
from utils.utils import pretty_print, penalty_contra, info_nce_loss, SimpleTransform, MyRotationTransform, GeneralizedCELoss
from dataset.dataset import get_10_corrupt_cifar_datasets_with_validate, our_dataset_cifar10
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='Corrupted Cifar10')
###whole settings
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--add_note', type=str, default="ccifar_ours")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--print_frequency_contra', type=int, default=100)
parser.add_argument('--save_epoch', type=int, default=80)
parser.add_argument('--eval_iter', type=int, default=5)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--dir_name', type=str, default="bias0.05")
parser.add_argument('--dir_root', type=str, default="./ccifar10")
parser.add_argument('--data_root', type=str, default="./data")
parser.add_argument('--save_dir', type=str, default="log")
###ours setting
parser.add_argument('--epochs_final', type=int, default=50)
parser.add_argument('--decay_final', type=int, default=30)
parser.add_argument('--lr_final_bias', type=float, default=0.02)
parser.add_argument('--contra_lr', type=float, default=0.002)
parser.add_argument('--stage1_epochs', type=int, default=800)
parser.add_argument('--iter_num', type=int, default=1)
parser.add_argument('--max_contra_num', type=int, default=60)
parser.add_argument('--max_complete_num', type=int, default=3)
parser.add_argument('--weight_base', type=float, default=1e-2) ###Note to adjust this in each setting! (1e-5, 1e-5, 5e-3, 1e-2) for (0.005 to 0.05)
parser.add_argument('--middle_mult', type=int, default=2)
parser.add_argument('--irm_weight', type=float, default=1.0)
parser.add_argument('--stage0_epochs', type=int, default=50)
parser.add_argument('--stage0_decay', type=int, default=30)
parser.add_argument('--select_rare_num', type=int, default=500)
parser.add_argument('--select_remain_num', type=int, default=7500)
parser.add_argument('--contra_dim', type=int, default=64)
parser.add_argument('--cls_weight', type=float, default=2.0)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size_contra', type=int, default=1000)


args, unknown = parser.parse_known_args()
additional_info = args.dir_name[4:]
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
test_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])
transform_contra = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(args.image_size, scale=(0.3, 1.0)),
    MyRotationTransform(angles=[0, 90, 180, 270]),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor()
])
print(args)

final_train_accs = []
final_test_accs_uniform = []
final_val_accs_uniform = []
def evaluate(model, loader,flag="", use_feature=False, model_contra=None):
    right_num = 0
    total_num = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        if use_feature:
            output = model_contra(data, need_contra_features=True).detach()
            output = model(output)
        else:
            output = model(data)
        y_pred = output.softmax(-1).argmax(-1)
        right_num += (y_pred == target).sum().cpu().item()
        total_num += data.size(0)
    print(flag,"accu:",right_num/ total_num)
    return right_num/ total_num
for restart in range(args.n_restarts):
    torch.cuda.empty_cache()
    print("Restart index:", restart)
    criterion_stage0 = nn.CrossEntropyLoss()
    gpu_list = [0, 1]
    gen_dataset = get_10_corrupt_cifar_datasets_with_validate(data_root=args.data_root,dir_root=args.dir_root, dir_name=args.dir_name)
    bias_criterion = GeneralizedCELoss()
    train_set = our_dataset_cifar10(gen_dataset.envs[0], train_transform)
    train_set_contra = our_dataset_cifar10(gen_dataset.envs[0], transform_contra,second_image=True)
    train_set_clear = our_dataset_cifar10(gen_dataset.envs[0], test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    train_loader_clear = torch.utils.data.DataLoader(
        train_set_clear, batch_size=args.batch_size, shuffle=False, num_workers=16)
    train_loader_contra = torch.utils.data.DataLoader(
        train_set_contra, batch_size=args.batch_size_contra, shuffle=True, num_workers=16, drop_last=True)
    ##training IRMCon (select samples. then train the IRMCon)
    for iter in range(args.iter_num):
        if iter == 0:
            model_stage0 = CNN_cifar(args).cuda()
            model_stage0 = torch.nn.DataParallel(model_stage0, device_ids=gpu_list)
            optimizer = optim.Adam(model_stage0.parameters(), 0.001)
        else:
            lin1 = nn.Linear(args.contra_dim, args.contra_dim*args.middle_mult)
            lin2 = nn.Linear(args.contra_dim*args.middle_mult, 10)
            for lin in [lin1, lin2]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            model_stage0 = nn.Sequential(lin1, nn.LeakyReLU(0.1),lin2).cuda()
            model_stage0 = torch.nn.DataParallel(model_stage0, device_ids=gpu_list)
            optimizer = optim.Adam(model_stage0.parameters(), lr=0.005)
        train_set.idx_flag = True
        for epoch in range(args.stage0_epochs):
            for batch_idx, (data, target, idx) in enumerate(train_loader):
                if epoch == args.stage0_decay and iter != 0:
                    optimizer.param_groups[0]['lr'] *= 0.1
                data, target = data.cuda(), target.cuda()
                if iter == 0:
                    output = model_stage0(data)
                    loss = criterion_stage0(output, target)
                else:
                    features_temp = model_contra(data, need_contra_features=True).detach()
                    output = model_stage0(features_temp)
                    loss = criterion_stage0(output, target)
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 25 == 0 and epoch != 0 or epoch == args.stage0_epochs - 1:
                print("stage0, train a model to assign weight for samples, epoch:", epoch, "loss:", loss.item())
        ##generate weights for samples
        if iter == 0:
            acc_train = evaluate(model_stage0, train_loader_clear, flag="train")
        else:
            acc_train = evaluate(model_stage0, train_loader_clear, flag="train", use_feature=True,
                                 model_contra=model_contra.eval())
        print("stage0 train accu{}".format(acc_train))
        train_set.idx_flag = False
        print("iter: {}, stage0 end".format(iter))
        estimated_weight_all = None
        train_set_clear.idx_flag = True
        for batch_idx, (data, target, idx) in enumerate(train_loader_clear):
            model_stage0.eval()
            data, target = data.cuda(), target.cuda()
            if iter == 0:
                output = model_stage0(data)
            else:
                features_temp = model_contra(data, need_contra_features=True).detach()
                output = model_stage0(features_temp)
            logits_soft = output.detach().softmax(-1)
            mask_gt = torch.zeros_like(logits_soft)
            mask_gt[range(mask_gt.size(0)), target] = 1
            estimated_weight = ((1 - mask_gt) * logits_soft).sum(-1)
            if estimated_weight_all is None:
                estimated_weight_all = estimated_weight
            else:
                estimated_weight_all = torch.cat((estimated_weight_all, estimated_weight),dim=0)
        train_set_clear.idx_flag = False
        print("iter {}, stage 1 (train IRMCon) start".format(iter))
        ##start training IRMCon
        model_contra = CNN_cifar(args,further_cls = True).cuda()
        model_contra = torch.nn.DataParallel(model_contra, device_ids=gpu_list)
        optimizer_contra = optim.Adam(model_contra.parameters(), lr=args.contra_lr)
        train_set_contra.weight_flag = True
        for epoch in range(args.stage1_epochs):
            if epoch == 600:
                optimizer_contra.param_groups[0]['lr'] *= 0.2
                args.cls_weight = args.cls_weight * 5
            ##sample from here, ramdom eliminate low weight samples
            sorted_weight, sorted_idx = torch.sort(estimated_weight_all,descending=True)
            total_num = estimated_weight_all.size(0)
            select_num = args.select_rare_num
            selected_idx = sorted_idx[:select_num]
            remain_idx = sorted_idx[select_num:]
            selected_idx2 = remain_idx[torch.randperm(remain_idx.size(0))][:args.select_remain_num]
            all_selected = torch.cat((selected_idx, selected_idx2), dim = 0)
            ###set all info of the dataset
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
                weight = weight.cuda()
                idx_selected_all = []
                sorted_weight, sorted_idx = torch.sort(weight, descending=False)
                idx_selected_all_cls = sorted_idx[:256]
                idx_selected_all_list = []
                ##set each enviroments
                for k in range(args.class_num):
                    list_temp = (target == k).nonzero(as_tuple=False).view(-1)
                    weight_temp = weight[list_temp]
                    select_value = torch.rand_like(weight_temp)
                    idx_intemp_list = (select_value * 0.8 < weight_temp).nonzero(as_tuple=False).view(-1)
                    ##shuffle
                    idx_intemp_list = idx_intemp_list[torch.randperm(idx_intemp_list.size(0))]
                    idx_intemp_list = idx_intemp_list[:args.max_contra_num]
                    idx_selected_temp = list_temp[idx_intemp_list]
                    idx_selected_all.append(idx_selected_temp)
                    idx_selected_all_list.extend(idx_selected_temp.tolist())
                len_all_list = len(idx_selected_all_list)
                select_final_idx = torch.Tensor(idx_selected_all_list).long()[
                    torch.randperm(len_all_list)[:args.max_contra_num]].cuda()
                idx_selected_all.append(select_final_idx)
                train_nll = 0
                train_penalty = 0
                for k in range(args.class_num+1):
                    temp_list = idx_selected_all[k]
                    temp_imgs1 = imgs1[temp_list]
                    temp_imgs2 = imgs2[temp_list]
                    if temp_imgs1.size(0) < 5:
                        continue
                    features1 = model_contra(temp_imgs1, need_contra_features=True)
                    features2 = model_contra(temp_imgs2, need_contra_features=True)
                    logits, labels = info_nce_loss(torch.cat((features1, features2), dim=0), features1.size(0),
                                                   temperature=1)
                    train_nll += torch.nn.CrossEntropyLoss()(logits, labels)
                    train_penalty += penalty_contra(logits, labels, torch.nn.CrossEntropyLoss()).mean()
                img_cls = imgs1[idx_selected_all_cls]
                label_cls = target[idx_selected_all_cls]
                logits_cls = model_contra(img_cls, need_contra_cls=True)
                cls_loss = nn.CrossEntropyLoss()(logits_cls, label_cls)
                train_nll = train_nll / float(args.class_num)
                train_penalty = train_penalty / float(args.class_num)
                loss = (train_nll + train_penalty * args.irm_weight) + args.cls_weight * cls_loss
                optimizer_contra.zero_grad()
                loss.backward()
                optimizer_contra.step()
            if (epoch % args.print_frequency_contra == 0 or epoch == args.stage1_epochs - 1) and epoch != 0:
                print("stage1, epoch:", epoch, "loss nll:", train_nll.item(),
                      "loss penalty:", train_penalty.item(), "loss cls:", cls_loss.item())
            if (epoch%args.save_epoch == 0 or epoch == args.stage1_epochs -1):
                save_subdir = "{}_bias{}".format(args.add_note, additional_info)
                save_path = os.path.join(args.save_dir, save_subdir)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = "starts_{}_final_model_ccontra_epoch{}.torch".format(restart,epoch)
                save_name = os.path.join(save_path, save_name)
                torch.save({'state': model_contra.state_dict()}, save_name)
    ##final training stage (IPW), like lff
    model_contra.eval()
    dataset_valid = our_dataset_cifar10(gen_dataset.envs[1], test_transform)
    dataset_test = our_dataset_cifar10(gen_dataset.envs[2], test_transform)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model_d =  ResNet18(args).cuda()
    gpu_list = [0, 1]
    model_d = torch.nn.DataParallel(model_d, device_ids=gpu_list)
    optimizer_d = optim.Adam(model_d.parameters(), lr=args.lr)

    lin0 = nn.Linear(args.contra_dim, 10)
    lin1 = nn.Linear(args.contra_dim, args.contra_dim * 12)
    lin2 = nn.Linear(args.contra_dim * 12, 10)
    for lin in [lin0, lin1, lin2]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
    model_b = nn.Sequential(lin1, nn.LeakyReLU(0.2), lin2).cuda()
    model_b = torch.nn.DataParallel(model_b, device_ids=gpu_list)
    optimizer_b = optim.Adam([{"params":model_b.parameters(), "lr":0.002}])

    criterion = nn.CrossEntropyLoss(reduction='none')
    best_acc = 0
    best_valid_acc = 0
    for epoch in range(args.epochs_final):
        if epoch == args.decay_final:
            optimizer_b.param_groups[0]['lr'] *= 0.1
            optimizer_d.param_groups[0]['lr'] *= 0.2
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.cuda()
            data = data.cuda()
            H_features_train = model_contra(data,need_contra_features=True)
            logit_b = model_b(H_features_train)
            loss_b = criterion(logit_b, target).cpu().detach()
            logit_d = model_d(data.cuda())
            loss_d = criterion(logit_d, target).cpu().detach()
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)  + args.weight_base
            loss_b_update = bias_criterion(logit_b, target)
            loss_d_update = criterion(logit_d, target) * loss_weight.cuda().detach()
            loss = loss_b_update.mean() + loss_d_update.mean()
            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()
        print("epoch num", epoch, "train loss:", loss.item())
        if (epoch % args.eval_iter == 0 or epoch == args.epochs_final - 1) and epoch != 0:
            model_b.eval()
            # can test the accuracy of bias model
            # acc_train = evaluate(model_b, train_loader_clear, flag="train", use_feature=True, model_contra=model_contra.eval())
            # acc_test = evaluate(model_b, test_loader, flag="test" , use_feature=True, model_contra=model_contra.eval())
            model_b.train()
            model_d.eval()
            train_acc = evaluate(model_d, train_loader_clear, flag="train")
            test_acc_valid = evaluate(model_d, valid_loader, flag="valid")
            test_acc_test = evaluate(model_d, test_loader, flag="test")
            print("epoch num", epoch, "train loss:", loss.item(),"train acc d:{} valid: {} test acc d:{}".format(train_acc,
                                                    test_acc_valid, test_acc_test))
            model_d.train()
            if test_acc_valid > best_valid_acc:
                best_acc = test_acc_test
                best_valid_acc = test_acc_valid
    final_train_accs.append(train_acc)
    final_test_accs_uniform.append(best_acc)
    final_val_accs_uniform.append(best_valid_acc)
    print("train:", round(np.mean(final_train_accs), 4), "test_uniform:",
          round(np.mean(final_test_accs_uniform), 4))
    print("accu(train mean-std):{}-{}"
          .format(round(np.mean(final_train_accs).item(), 4), round(np.std(final_train_accs).item(), 4)))
    print("accu(valid (best selected by valid) uniform mean-std):{}-{}"
          .format(round(np.mean(final_val_accs_uniform).item(), 4),
                  round(np.std(final_val_accs_uniform).item(), 4)))
    print("accu(test (best selected by valid) uniform mean-std):{}-{}"
          .format(round(np.mean(final_test_accs_uniform).item(), 4),
                  round(np.std(final_test_accs_uniform).item(), 4)))
    ##save models
    save_subdir = "{}_bias{}".format(args.add_note, additional_info)
    save_path = os.path.join(args.save_dir, save_subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = "starts_{}_final_modelb.torch".format(restart)
    save_name = os.path.join(save_path, save_name)
    torch.save({'state': model_b.state_dict()}, save_name)
    save_name = "starts_{}_final_modeld.torch".format(restart)
    save_name = os.path.join(save_path, save_name)
    torch.save({'state': model_d.state_dict()}, save_name)



