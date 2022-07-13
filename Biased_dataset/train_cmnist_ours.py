import argparse
import numpy as np
import torch
import os
from torch import nn, optim
from models.model_ours import Linear_mnist
from utils.utils import pretty_print, penalty_contra, info_nce_loss, SimpleTransform, GeneralizedCELoss
from dataset.dataset import get_10_color_mnist_datasets_with_validate, our_dataset_cmnist
parser = argparse.ArgumentParser(description='Colored MNIST')
###base settings
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--add_note', type=str, default="cmnist_ours")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--train_print_interval', type=int, default=100)
parser.add_argument('--eval_iter', type=int, default=5)
parser.add_argument('--save_epoch', type=int, default=800)
parser.add_argument('--dir_name', type=str, default="bias0.05")
parser.add_argument('--dir_root', type=str, default="./cmnist")
parser.add_argument('--data_root', type=str, default="./data")
parser.add_argument('--save_dir', type=str, default="log")
###ours setting
parser.add_argument('--epochs_final', type=int, default=50)
parser.add_argument('--decay_final_unbias', type=int, default=30)
parser.add_argument('--decay_final_bias', type=int, default=10)
parser.add_argument('--lr_final_bias', type=float, default=0.02)
parser.add_argument('--contra_lr', type=float, default=0.008)
parser.add_argument('--stage1_epochs', type=int, default=400)
parser.add_argument('--iter_num', type=int, default=1)
parser.add_argument('--max_contra_num', type=int, default=50)
parser.add_argument('--max_complete_num', type=int, default=3)
parser.add_argument('--weight_base', type=float, default=1e-3) ###Note to adjust this in each setting!  (2e-5, 2e-5, 2e-5, 1e-4, 2e-4, 1e-3) for (0.001 to 0.05)
parser.add_argument('--final_dim', type=int, default=12) ##context dimension
parser.add_argument('--middle_mult', type=int, default=8)
parser.add_argument('--irm_weight', type=float, default=1.0)
parser.add_argument('--stage0_epochs', type=int, default=100)
parser.add_argument('--stage0_decay', type=int, default=60)

args, unknown = parser.parse_known_args()
additional_info = args.dir_name[4:]
augmentation = SimpleTransform(image_size = 28)
print(args)

final_train_accs = []
final_test_accs_uniform = []
final_val_accs_uniform = []

def evaluate(model, loader, flag="", features=None):
    right_num = 0
    total_num = 0
    for batch_idx, (data, target, idx) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        if features is None:
            output = model(data)
        else:
            output = model(features[idx.cuda()])
        y_pred = output.softmax(-1).argmax(-1)
        right_num += (y_pred == target).sum().cpu().item()
        total_num += data.size(0)
    print(flag, "accu:", right_num / total_num)
    return right_num / total_num
for restart in range(args.n_restarts):
    torch.cuda.empty_cache()
    print("Restart index:", restart)
    ##mute at second time
    gpu_list = [0, 1]
    gen_dataset = get_10_color_mnist_datasets_with_validate(data_root = args.data_root, dir_root=args.dir_root, dir_name=args.dir_name)
    envs_origin = gen_dataset.envs
    bias_criterion = GeneralizedCELoss()
    ##training IRMCon (select samples. then train the IRMCon)
    for iter in range(args.iter_num):
        env_temp_origin = envs_origin[0]
        envs = envs_origin
        envs_stage1 = [envs_origin[0]]
        if iter == 0:
            model_stage0 = Linear_mnist().cuda()
            model_stage0 = torch.nn.DataParallel(model_stage0, device_ids=gpu_list)
            optimizer = optim.Adam(model_stage0.parameters(), lr=0.002)
        else:
            lin1 = nn.Linear(args.final_dim, args.final_dim*args.middle_mult)
            lin2 = nn.Linear(args.final_dim*args.middle_mult, 10)
            for lin in [lin1, lin2]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            model_stage0 = nn.Sequential(lin1, nn.LeakyReLU(0.1),lin2).cuda()
            model_stage0 = torch.nn.DataParallel(model_stage0, device_ids=gpu_list)
            optimizer = optim.Adam(model_stage0.parameters(), lr=0.01)
        if iter == 0:
            model_contra = Linear_mnist(final_dim=args.final_dim).cuda()
            model_contra = torch.nn.DataParallel(model_contra, device_ids=gpu_list)
        model_contra.train()
        optimizer_contra = optim.Adam(model_contra.parameters(), lr=args.contra_lr)

        for epoch in range(args.stage0_epochs):
            if epoch == args.stage0_decay and iter != 0:
                optimizer.param_groups[0]['lr'] *= 0.1
            for idx, env in enumerate(envs_stage1):
                curent_size = env['images'].shape[0]
                env['size'] = curent_size
                if iter == 0:
                    logits = model_stage0(env['images'])
                else:
                    logits = model_stage0(env['features'])
                env["logits"] = logits
                env['nll'] = bias_criterion(logits, env['labels']).mean()
            train_nll = envs_stage1[0]['nll']
            loss = train_nll.clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model_stage0.eval()
        if iter == 0:
            logits = model_stage0(env['images'])
        else:
            logits = model_stage0(env['features'])
        ###tricks select samples by weights
        logits_soft = logits.detach().softmax(-1)
        mask_gt = torch.zeros_like(logits_soft)
        mask_gt[range(mask_gt.size(0)), envs_stage1[0]['labels']] = 1
        right_exp = (mask_gt * logits_soft).sum(-1)
        wrong_exp = ((1 - mask_gt) * logits_soft).sum(-1)
        weights = wrong_exp
        envs_stage1[0]["weights"] = weights
        idx_list = []
        for k in range(10):
            idx_temp = (envs_stage1[0]['labels'] == k).nonzero(as_tuple=False).view(-1)
            idx_list.append(idx_temp)
        envs_stage1[0]["idx_list"] = idx_list
        pretty_print('epoch', 'train loss', 'irm loss')
        for epoch in range(args.stage1_epochs):
            if epoch % 10 == 0: ##sample H
                idx_selected_all = []
                for k in range(10):
                    ###change a selection strategy
                    idx_selected_temp = []
                    list_temp = envs_stage1[0]["idx_list"][k]
                    weight_temp = envs_stage1[0]["weights"][list_temp]
                    ######### select method:,
                    select_value = torch.rand_like(weight_temp)
                    idx_intemp_list = (select_value < weight_temp).nonzero(as_tuple=False).view(-1)
                    ###shuffle
                    idx_intemp_list = idx_intemp_list[torch.randperm(idx_intemp_list.size(0))]
                    idx_intemp_list = idx_intemp_list[:args.max_contra_num]
                    ####  random select others for complement
                    additional_list = torch.randperm(weight_temp.size(0))[:args.max_complete_num].cuda()
                    idx_intemp_list = torch.cat((idx_intemp_list,additional_list ),dim=0)
                    ###final selection
                    idx_selected_temp = list_temp[idx_intemp_list]
                    idx_selected_all.append(idx_selected_temp)
                envs_stage1[0]["idx_selected"] = idx_selected_all
                env_num = 10
                ############################################################over
                new_envs = []
                if iter == 0:
                    key_list =["images", "labels"]
                else:
                    key_list =["images", "labels","features"]
                for k in range(env_num):
                    t_env = {}
                    for key in key_list:
                        t_env[key] = envs_stage1[0][key][envs_stage1[0]["idx_selected"][k].long()]
                    t_env["idx_selected"] = envs_stage1[0]["idx_selected"][k]
                    new_envs.append(t_env)
            ####ours IRMCon
            for idx, env in enumerate(new_envs):
                curent_size = env['images'].shape[0]
                env['size'] = curent_size
                images1, images2 = augmentation.transform_batch(env['images'].permute(0, 3, 1, 2))
                features1 = model_contra.module.get_features_contra(images1.permute(0, 2, 3, 1))
                features2 = model_contra.module.get_features_contra(images2.permute(0, 2, 3, 1))
                logits, labels = info_nce_loss(torch.cat((features1, features2), dim=0), features1.size(0),
                                               temperature=0.5,
                                               normalize=False)
                train_nll = torch.nn.CrossEntropyLoss()(logits, labels)
                env["logits"] = logits
                env['penalty'] = penalty_contra(logits, labels, torch.nn.CrossEntropyLoss()).mean()
                env['nll'] = train_nll
            all_penalty = []
            all_null = []
            all_acc = []
            for p in range(0, len(new_envs)):
                all_penalty.append(new_envs[p]['penalty'])
                all_null.append(new_envs[p]['nll'])
            train_penalty = torch.stack(all_penalty).mean()
            train_nll = torch.stack(all_null).mean()
            loss = train_nll.clone() + train_penalty * args.irm_weight
            optimizer_contra.zero_grad()
            loss.backward()
            optimizer_contra.step()
            if epoch % args.train_print_interval == 0:
                pretty_print(
                    np.int32(epoch),
                    train_nll.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                )
            if (epoch % args.save_epoch == 0 or epoch == args.stage1_epochs - 1):
                save_subdir = "{}_bias{}".format(args.add_note, additional_info)
                save_path = os.path.join(args.save_dir, save_subdir)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = "starts_{}_final_model_contra_epoch{}_iter{}.torch".format(restart, epoch, iter)
                save_name = os.path.join(save_path, save_name)
                torch.save({'state': model_contra.state_dict()}, save_name)
        ##extracting context features
        env_num = 0
        model_contra.eval()
        features_biased = model_contra.module.get_features_contra(envs_stage1[env_num]["images"]).detach().cpu().numpy()
        envs_stage1[env_num]["features"] = torch.Tensor(features_biased)
        model_contra.train()
    model_contra.eval()
    H_features_train = model_contra.module.get_features_contra(envs[0]["images"]).detach().cpu()
    H_features_test = model_contra.module.get_features_contra(envs[2]["images"]).detach().cpu()

    ####final training stage (IPW), like lff
    dataset_train = our_dataset_cmnist(gen_dataset.envs[0])
    dataset_valid = our_dataset_cmnist(gen_dataset.envs[1])
    dataset_test = our_dataset_cmnist(gen_dataset.envs[2])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_loader_eval = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model_d = Linear_mnist().cuda()
    gpu_list = [0, 1]
    model_d = torch.nn.DataParallel(model_d, device_ids=gpu_list)
    optimizer_d = optim.Adam(model_d.parameters(), lr=args.lr)

    lin1 = nn.Linear(args.final_dim, args.final_dim * args.middle_mult)
    lin2 = nn.Linear(args.final_dim * args.middle_mult, 10)
    for lin in [lin1, lin2]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
    model_b = nn.Sequential(lin1, nn.LeakyReLU(0.1), lin2).cuda()
    model_b = torch.nn.DataParallel(model_b, device_ids=gpu_list)
    optimizer_b = optim.Adam(model_b.parameters(), lr=args.lr_final_bias)

    criterion = nn.CrossEntropyLoss(reduction='none')
    best_acc = 0
    best_valid_acc = 0
    dataset_train.idx_flag = True
    dataset_test.idx_flag = True
    dataset_valid.idx_flag = True
    for epoch in range(args.epochs_final):
        if epoch == args.decay_final_bias:
            optimizer_b.param_groups[0]['lr'] *= 0.2
        if epoch == args.decay_final_unbias:
            optimizer_d.param_groups[0]['lr'] *= 0.2
        for batch_idx, (data, target, idx_temp) in enumerate(train_loader):
            target = target.cuda()
            logit_b = model_b(H_features_train[idx_temp.cuda()])
            loss_b = criterion(logit_b, target).cpu().detach()
            logit_d = model_d(data.cuda())  ##all input c,h,w
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
        if (epoch % args.eval_iter == 0 or epoch == args.epochs_final - 1) and epoch != 0:
            model_b.eval()
            train_acc_b = evaluate(model_b, train_loader_eval, flag="train", features = H_features_train)
            test_acc_b = evaluate(model_b, test_loader, flag="test", features = H_features_test)
            model_b.train()
            model_d.eval()
            train_acc = evaluate(model_d, train_loader_eval, flag="train")
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
    print("train:", round(np.mean(final_train_accs), 4), "test_uniform:", round(np.mean(final_test_accs_uniform), 4))
    print("accu(train mean-std):{}-{}"
          .format(round(np.mean(final_train_accs).item(), 4), round(np.std(final_train_accs).item(), 4)))
    print("accu(valid (best selected by valid) uniform mean-std):{}-{}"
          .format(round(np.mean(final_val_accs_uniform).item(), 4), round(np.std(final_val_accs_uniform).item(), 4)))
    print("accu(test uniform mean-std):{}-{}"
          .format(round(np.mean(final_test_accs_uniform).item(), 4), round(np.std(final_test_accs_uniform).item(), 4)))
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



