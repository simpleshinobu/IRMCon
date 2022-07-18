import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
sys.path.append("../")
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


def set_training_aug_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.training_aug = value
    temp_set.datasets[1].underlying_dataset.training_aug = value
    temp_set.datasets[2].underlying_dataset.training_aug = value


def set_training_dataset_idx_flag(temp_set, value):
    temp_set.datasets[0].underlying_dataset.idx_flag = value
    temp_set.datasets[1].underlying_dataset.idx_flag = value
    temp_set.datasets[2].underlying_dataset.idx_flag = value
    temp_set.datasets[0].underlying_dataset.dataset_idx = value
    temp_set.datasets[1].underlying_dataset.dataset_idx = value
    temp_set.datasets[2].underlying_dataset.dataset_idx = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str,default="./data/")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int,default=128)
    parser.add_argument('--workers', type=int,default=8)
    parser.add_argument('--add_note', type=str,default="test")
    parser.add_argument('--lr', type=float,default=0.001)
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--n_start', type=int, default=3)
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=4800,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=1) ##per epochs
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    final_test_accs = []

    for start in range(args.n_start):
        start_step = 0
        algorithm_dict = None
        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                      misc.seed_hash(args.hparams_seed, args.trial_seed))
        if args.hparams:
            hparams.update(json.loads(args.hparams))
        print("lr", hparams["lr"])
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        if args.lr != -1:
            hparams["lr"] = args.lr
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

            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

            if env_i in args.test_envs:
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*args.uda_holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))

        if args.task == "domain_adaptation" and len(uda_splits) == 0:
            raise ValueError("Not enough unlabeled samples for domain adaptation.")


        from torch.utils.data import ConcatDataset
        train_set_list = []
        for i, (dataset_temp, _) in enumerate(in_splits):
            if i not in args.test_envs:
                train_set_list.append(dataset_temp)
        train_split_set = ConcatDataset(train_set_list)
        train_loader = torch.utils.data.DataLoader(
            train_split_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print("test env", args.test_envs)
        if args.dataset == "PACS":
            print("test env number 0,1,2,3 for ['art_painting', 'cartoon', 'photo', 'sketch']")

        eval_test_set = out_splits[args.test_envs[0]][0]
        eval_train_list = []
        for i, (dataset_temp, _) in enumerate(out_splits):
            if i not in args.test_envs:
                eval_train_list.append(dataset_temp)
        eval_train_set = ConcatDataset(eval_train_list)
        ###make final test
        real_test_set = in_splits[args.test_envs[0]][0]
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.workers)
            for env in [train_split_set, eval_train_set, eval_test_set, real_test_set]]

        eval_loader_names = ["0.train", "1.eval(train)","2.eval(test)","3.test"]

        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        algorithm.to(device)

        checkpoint_vals = collections.defaultdict(lambda: [])

        steps_per_epoch = 48
        n_steps = args.steps
        checkpoint_freq = args.checkpoint_freq

        def save_checkpoint(filename):
            if args.skip_model_save:
                return
            save_dict = {
                "args": vars(args),
                "model_input_shape": dataset.input_shape,
                "model_num_classes": dataset.num_classes,
                "model_num_domains": len(dataset) - len(args.test_envs),
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict()
            }
            torch.save(save_dict, os.path.join(args.output_dir, filename))


        last_results_keys = None
        best_accs = {"best_eval1": 0, "best_test1": 0, "best_eval2": 0, "best_test2": 0, "best_iters1": 0,
                     "best_iters2": 0}
        uda_device = None
        best_acc_val = 0
        best_acc_test = 0

        for step in range(0, args.epochs):
            step_start_time = time.time()
            set_training_dataset_idx_flag(train_split_set, True)
            set_training_aug_flag(train_split_set, True)
            for batch_idx, (x, y, _,  dataset_idx) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                minibatches_device = []
                for k in range(4):
                    temp_idx = (dataset_idx == k).nonzero(as_tuple=False).view(-1)
                    if temp_idx.size(0) == 0:
                        pass
                    else:
                        temp_tuple = (x[temp_idx],y[temp_idx])
                        minibatches_device.append(temp_tuple)
                step_vals = algorithm.update(minibatches_device, uda_device)


            set_training_dataset_idx_flag(train_split_set, False)
            set_training_aug_flag(train_split_set, False)
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            if (step % args.checkpoint_freq == 0) or (step == args.epochs - 1):
                results = {'epoch': step}
                set_training_aug_flag(train_split_set, False)
                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)
                weights_all = [None, None, None, None]
                evals = zip(eval_loader_names, eval_loaders, weights_all)
                accu_temp_list = []
                for i, (name, loader, weights) in enumerate(evals):
                    acc = misc.accuracy(algorithm, loader, weights, device)
                    results[name+'_acc'] = acc
                    accu_temp_list.append(acc)
                #select model and print selected accu in test
                if accu_temp_list[1] > best_accs["best_eval1"]:
                    best_accs["best_eval1"] = accu_temp_list[1]
                    best_accs["best_test1"] = accu_temp_list[-1]
                    best_accs["best_iters1"] = step
                    save_checkpoint('model_select_from_train.pkl')
                if accu_temp_list[2] > best_accs["best_eval2"]:
                    best_accs["best_eval2"] = accu_temp_list[2]
                    best_accs["best_test2"] = accu_temp_list[-1]
                    best_accs["best_iters2"] = step
                    save_checkpoint('model_select_from_test.pkl')

                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            if (step % 10 == 0) or (step == args.epochs - 1):
                #log results, no change
                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                               colwidth=12)
                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })
                print("best accu selected from train domain at {}: #### {} ####; best accu selected from test domain at {}: #### {} ####"
                      .format(best_accs["best_iters1"],best_accs["best_test1"] ,best_accs["best_iters2"],best_accs["best_test2"] ))

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
        save_checkpoint('model_final_{}.pkl'.format(start))

        final_test_accs.append(best_accs["best_test2"])

        print("accu(test selected from test uniform mean-std):{}-{}"
              .format(round(np.mean(final_test_accs).item(), 4),
                      round(np.std(final_test_accs).item(), 4)))
        ##save models
        # with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        #     f.write('done')