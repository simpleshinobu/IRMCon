import argparse
import numpy as np
import torch
import os
from torch import nn, optim
from models.model_ours import Linear_mnist
from dataset.dataset import get_10_color_mnist_datasets_with_validate, our_dataset_cmnist

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--decay_step', type=int, default=30)
parser.add_argument('--eval_iter', type=int, default=5)

parser.add_argument('--num_colors', type=int, default=10)
parser.add_argument('--add_note', type=str, default="cmnist_erm")
parser.add_argument('--dir_name', type=str, default="bias0.05")
parser.add_argument('--dir_root', type=str, default="./cmnist")
parser.add_argument('--data_root', type=str, default="./data")
parser.add_argument('--save_dir', type=str, default="log")


args, unknown = parser.parse_known_args()
additional_info = args.dir_name[4:]
print(args)

final_train_accs = []
final_test_accs_uniform = []
final_val_accs_uniform = []

def evaluate(model, loader,flag=""):
    right_num = 0
    total_num = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        y_pred = output.softmax(-1).argmax(-1)
        right_num += (y_pred == target).sum().cpu().item()
        total_num += data.size(0)
    print(flag,"accu:",right_num/ total_num)
    return right_num/ total_num

for restart in range(args.n_restarts):
    torch.cuda.empty_cache()
    print("Restart index:", restart)
    gen_dataset = get_10_color_mnist_datasets_with_validate(data_root = args.data_root ,dir_root=args.dir_root, dir_name=args.dir_name)
    dataset_train = our_dataset_cmnist(gen_dataset.envs[0])
    dataset_train_eval = our_dataset_cmnist(gen_dataset.envs[0])
    dataset_valid = our_dataset_cmnist(gen_dataset.envs[1])
    dataset_test = our_dataset_cmnist(gen_dataset.envs[2])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_loader_eval = torch.utils.data.DataLoader(
        dataset_train_eval, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)
    ##set models
    model = Linear_mnist().cuda()
    gpu_list = [0, 1]
    model = torch.nn.DataParallel(model, device_ids=gpu_list)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    best_valid_acc = 0
    for step in range(args.steps):
        if step == args.decay_step:
            optimizer.param_groups[0]['lr'] *= 0.2
        for batch_idx, (data, target) in enumerate(train_loader):
            logits = model(data.cuda()) ##all input c,h,w
            loss = nn.CrossEntropyLoss()(logits, target.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch num", step, "train loss:", loss.item())
        if (step % args.eval_iter == 0 or step == args.steps -1) and step != 0:
            model.eval()
            train_acc = evaluate(model, train_loader_eval, flag="train")
            test_acc_valid = evaluate(model, valid_loader, flag="valid")
            test_acc_test = evaluate(model, test_loader, flag="test")
            model.train()
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
    print("accu(test (best selected by valid) uniform mean-std):{}-{}"
          .format(round(np.mean(final_test_accs_uniform).item(), 4), round(np.std(final_test_accs_uniform).item(), 4)))
    ##save models
    save_subdir = "{}_bias{}".format(args.add_note, additional_info)
    save_path = os.path.join(args.save_dir, save_subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = "starts_{}_final_model.torch".format(restart)
    save_name = os.path.join(save_path, save_name)
    torch.save({'state': model.state_dict()}, save_name)


