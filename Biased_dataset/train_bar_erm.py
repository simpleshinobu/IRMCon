import torchvision.transforms as transforms
import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch import  optim
from dataset.dataset import our_dataset_bar, split_test
from utils.utils import cosine_schedule
from models.backbones import ResNet18
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='BAR')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_interval', type=int, default=5)
parser.add_argument('--train_print_interval', type=int, default=5)
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--decay_step', type=int, default=160)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--warm_epochs', type=int, default=10)
parser.add_argument('--eval_epoch', type=int, default=200)
parser.add_argument('--add_note', type=str, default="bar_erm")
parser.add_argument('--save_dir', type=str, default="log")
parser.add_argument('--schedule', type=str, default="cosine")
parser.add_argument('--ratio', type=float, default=0.05)
args, unknown = parser.parse_known_args()

def target_transform(target):
    return int(target)
def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, lr):
    p = (batch_id + 1 + epoch * total_batches) / (args.warm_epochs * total_batches)
    lr = p * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
from randaugment import RandAugment
transform = transforms.Compose([
    transforms.RandomResizedCrop(args.image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.Resize(int(args.image_size*1.1)),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    normalize
])

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

    split_test(args)
    train_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="train",
                                transform=transform, save_flag=True)
    train_set_clear = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="train",
                                      transform=test_transform, save_flag=True)
    valid_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="valid",
                                transform=test_transform, save_flag=True)
    test_set = our_dataset_bar(data_path="./data/BAR/dataset{}_temp.torch".format(args.ratio), set_name="test",
                               transform=test_transform, save_flag=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_loader_clear = torch.utils.data.DataLoader(
        train_set_clear, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    acc_test_best = 0
    torch.cuda.empty_cache()
    print("Restart index:", restart)
    model = ResNet18().cuda()
    gpu_list = [0, 1]
    model = torch.nn.DataParallel(model, device_ids=gpu_list)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    best_valid_acc = 0
    args.warmup_from = 0.
    args.warmup_to = args.lr
    for epoch in range(args.epochs):
        if epoch == args.decay_step and args.schedule == "step":
            optimizer.param_groups[0]['lr'] *= 0.2
        elif epoch > args.warm_epochs and args.schedule == "cosine":
            cosine_schedule(optimizer, epoch, args, args.lr)
        for batch_idx, (data, target) in enumerate(train_loader):
            if epoch < args.warm_epochs:
                warmup_learning_rate(optimizer, epoch, batch_idx, len(train_loader), args.lr)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch",epoch,"loss",loss,"lr",optimizer.param_groups[0]['lr'])
        test_interval = args.test_interval
        if (epoch % args.test_interval == 0 and epoch !=0 or epoch == args.epochs - 1):
            print("epochs:", epoch)
            model.eval()
            train_acc = evaluate(model, train_loader_clear, flag="train")
            test_acc_valid = evaluate(model, valid_loader, flag="valid")
            test_acc_test = evaluate(model, test_loader, flag="test")
            model.train()
            if epoch > args.eval_epoch:
                if test_acc_valid > best_valid_acc:
                    best_acc = test_acc_test
                    best_valid_acc = test_acc_valid
                    save_subdir = "bs_{}_epochs_{}_{}".format(args.batch_size, args.epochs, args.add_note)
                    save_path = os.path.join(args.save_dir, save_subdir)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_name = "starts_{}_final_model_d_best.torch".format(restart)
                    save_name = os.path.join(save_path, save_name)
                    torch.save({'state': model.state_dict()}, save_name)
                    print("best acc{}:".format(best_acc), "save model into {}".format(save_name))

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
    save_subdir = "bs_{}_epochs_{}_{}".format(args.batch_size, args.epochs, args.add_note)
    save_path = os.path.join(args.save_dir, save_subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = "starts_{}_final_model.torch".format(restart)
    save_name = os.path.join(save_path, save_name)
    torch.save({'state': model.state_dict()}, save_name)



