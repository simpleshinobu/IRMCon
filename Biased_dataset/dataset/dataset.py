import torch
import numpy as np
import os
from PIL import Image

class get_10_color_mnist_datasets_with_validate(object):
    def __init__(self,data_root = "",dir_root = ""
    ,dir_name = ""):
        super(get_10_color_mnist_datasets_with_validate, self).__init__()
        self.envs = []
        self.train_envs = []
        self.test_envs = []
        split_ratio = 0.2
        data_path = os.path.join(data_root, dir_root)
        data_path = os.path.join(data_path,dir_name)
        train_images = np.load(os.path.join(data_path,"train/images.npy"))/255.0
        test_images_raw = np.load(os.path.join(data_path,"valid/images.npy"))/255.0
        test_labels_raw = np.load(os.path.join(data_path,"valid/attrs.npy"))[:,0]
        test_colors_raw = np.load(os.path.join(data_path,"valid/attrs.npy"))[:,1]

        test_size = test_labels_raw.shape[0]
        select_valid_size = int(test_size * split_ratio)
        rand_idx = torch.randperm(test_size)

        valid_images = test_images_raw[rand_idx[:select_valid_size]]
        valid_labels = test_labels_raw[rand_idx[:select_valid_size]]
        valid_colors = test_colors_raw[rand_idx[:select_valid_size]]

        test_images = test_images_raw[rand_idx[select_valid_size:]]
        test_labels = test_labels_raw[rand_idx[select_valid_size:]]
        test_colors = test_colors_raw[rand_idx[select_valid_size:]]

        train_labels = np.load(os.path.join(data_path,"train/attrs.npy"))[:,0]
        train_labels_color = np.load(os.path.join(data_path,"train/attrs.npy"))[:,1]
        list_img = [train_images, valid_images, test_images]
        list_label = [train_labels, valid_labels, test_labels]
        list_label_color = [train_labels_color, valid_colors, test_colors]
        total_range = len(list_img)
        for k in range(total_range):
            env = {}
            env["images"] = torch.Tensor(list_img[k]).cuda()
            env["labels"] = torch.Tensor(list_label[k]).long().cuda()
            env["colors"] = torch.Tensor(list_label_color[k]).long().cuda()
            self.envs.append(env)


class get_10_corrupt_cifar_datasets_with_validate(object):
    def __init__(self,data_root = "",dir_root = ""
    ,dir_name = ""):
        super(get_10_corrupt_cifar_datasets_with_validate, self).__init__()
        split_ratio = 0.2
        self.envs = []
        self.train_envs = []
        self.test_envs = []
        valid_num = 1
        data_path = os.path.join(data_root, dir_root)
        data_path = os.path.join(data_path,dir_name)
        train_images = np.load(os.path.join(data_path,"train/images.npy"))/255.0
        train_labels = np.load(os.path.join(data_path, "train/attrs.npy"))[:,0]
        train_labels_color = np.load(os.path.join(data_path, "train/attrs.npy"))[:,1]
        test_images_raw = np.load(os.path.join(data_path,"valid/images.npy"))/255.0
        test_labels_raw = np.load(os.path.join(data_path, "valid/attrs.npy"))[:,0]
        test_labels_raw_color = np.load(os.path.join(data_path, "valid/attrs.npy"))[:,1]

        test_size = test_labels_raw.shape[0]
        select_valid_size = int(test_size * split_ratio)
        rand_idx = torch.randperm(test_size)

        valid_images = test_images_raw[rand_idx[:select_valid_size]]
        valid_labels = test_labels_raw[rand_idx[:select_valid_size]]
        valid_labels_color = test_labels_raw_color[rand_idx[:select_valid_size]]

        test_images = test_images_raw[rand_idx[select_valid_size:]]
        test_labels = test_labels_raw[rand_idx[select_valid_size:]]
        test_labels_color = test_labels_raw_color[rand_idx[select_valid_size:]]
        list_img = [train_images, valid_images, test_images]
        list_label = [train_labels, valid_labels, test_labels]
        list_label_color = [train_labels_color, valid_labels_color, test_labels_color]

        total_range = len(list_img)

        for k in range(total_range):
            env = {}
            env["images"] = torch.Tensor(list_img[k]).cuda()
            env["labels"] = torch.Tensor(list_label[k]).long().cuda()
            env["colors"] = torch.Tensor(list_label_color[k]).long().cuda()
            self.envs.append(env)

class our_dataset_cifar10:
    def __init__(self, env, transform, second_image=False):

        self.image_list = env["images"].cpu().permute(0, 3, 1, 2)
        self.label_list = env["labels"].cpu()
        self.attr_list = env["colors"].cpu()
        self.weight_list = []
        self.weight_flag = False
        self.idx_flag = False
        self.get_color_flag = False
        self.transform = transform
        self.second_image = second_image

    def __getitem__(self, i):
        img_origin = self.image_list[i]
        img = self.transform(img_origin)
        if self.second_image:
            img2 = self.transform(img_origin)
            img = (img, img2)
        target = self.label_list[i]
        if self.weight_flag:
            weight = self.weight_list[i]
            return img, target, weight
        if self.idx_flag:
            return img, target, i
        if self.get_color_flag:
            return img, target, self.attr_list[i]
        return img, target

    def __len__(self):
        return len(self.image_list)

class our_dataset_cmnist:
    def __init__(self, env, transform=None, second_image=False):

        self.image_list = env["images"].cpu()
        self.label_list = env["labels"].cpu()
        self.attr_list = env["colors"].cpu()
        self.weight_list = []
        self.weight_flag = False
        self.idx_flag = False
        self.transform = transform
        self.second_image = second_image
        self.get_color_flag = False
    def __getitem__(self, i):
        img_origin = self.image_list[i]
        if self.transform is not None:
            img = self.transform(img_origin)
        else:
            img = img_origin
        if self.second_image:
            img2 = self.transform(img_origin)
            img = (img, img2)
        target = self.label_list[i]
        if self.weight_flag:
            weight = self.weight_list[i]
            return img, target, weight
        if self.idx_flag:
            return img, target, i
        if self.get_color_flag:
            return img, target, self.attr_list[i]
        return img, target

    def __len__(self):
        return len(self.image_list)


def split_test(args):
    print("spliting dataset ... 2/8")
    datas = torch.load("./data/BAR/dataset{}.torch".format(args.ratio))
    split_ratio = 0.2
    test_image_list = datas["test"][0]
    test_label_list = datas["test"][1]
    test_size = len(datas["test"][0])
    select_valid_size = int(test_size * split_ratio)
    rand_idx = torch.randperm(select_valid_size).tolist()
    valid_list_img_new = []
    test_list_img_new = []
    valid_list_label_new = []
    test_list_label_new = []
    for k in range(test_size):
        if k in rand_idx:
            valid_list_img_new.append(test_image_list[k])
            valid_list_label_new.append(test_label_list[k])
        else:
            test_list_img_new.append(test_image_list[k])
            test_list_label_new.append(test_label_list[k])
    import copy
    datas_new = {}
    datas_new["train"] = datas["train"]
    datas_new["valid"] = [valid_list_img_new, valid_list_label_new]
    datas_new["test"] = [test_list_img_new, test_list_label_new]
    datas_new["label_info"] = datas["label_info"]
    torch.save(datas_new, "./data/BAR/dataset{}_temp.torch".format(args.ratio))

class our_dataset_bar:
    def __init__(self, data_path = "./data/BAR/dataset.torch", set_name = "train"
                 , transform = None, save_flag = False, second_image=False, transform_easy = None):
        datasets = torch.load(data_path)
        self.image_list = datasets[set_name][0]
        self.label_list = datasets[set_name][1]
        self.weight_list = []
        self.weight_flag = False
        self.idx_flag = False
        self.transform = transform
        self.transform_easy = transform_easy
        self.save_flag = save_flag
        self.second_image = second_image
        self.return_origin_flag = False
        if save_flag:
            self.images = [None] * len(self.image_list)
    def __getitem__(self, i):
        image_path = "./data/" + self.image_list[i]
        img_origin = Image.open(image_path).convert('RGB')
        img = self.transform(img_origin)
        if self.return_origin_flag:
            img_origin = self.transform_easy(img_origin)
            img = (img, img_origin)
        if self.second_image:
            img2 = self.transform(img_origin)
            img = (img, img2)
        target = self.label_list[i]
        if self.weight_flag:
            weight = self.weight_list[i]
            return img, target, weight
        if self.idx_flag:
            return img, target, i
        return img, target

    def __len__(self):
        return len(self.image_list)