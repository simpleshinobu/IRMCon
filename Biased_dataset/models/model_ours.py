from torch import nn
class Linear_mnist(nn.Module):
    def __init__(self,input_size =3 * 28 * 28, final_dim=100):
        super(Linear_mnist, self).__init__()
        hidden_dim = 100
        lin1 = nn.Linear(input_size, hidden_dim)
        self.imput_size = input_size
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, hidden_dim)
        lin4 = nn.Linear(hidden_dim, 10)
        lin_contra_f = nn.Linear(hidden_dim, final_dim)
        self.final_cls = lin4
        self._features = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3, nn.ReLU())
        self.final_contra_cls_f = lin_contra_f
    def forward(self, input_imgs, need_features = False):
        input_imgs = input_imgs.permute(0, 3, 1, 2).contiguous().view(-1,self.imput_size)
        features = self._features(input_imgs)
        out = self.final_cls(features)
        if need_features:
            return features, out
        return out
    def get_features_contra(self, input_imgs):
        input_imgs = input_imgs.permute(0,3,1,2).contiguous().view(-1,self.imput_size)
        features = self._features(input_imgs)
        features = self.final_contra_cls_f(features)
        return features

class CNN_cifar(nn.Module):
    def __init__(self,args=None,input_channel = 3, hidden_channel = 32,further_cls = False):
        super(CNN_cifar, self).__init__()
        conv1 = nn.Conv2d(input_channel, hidden_channel, kernel_size= 3, stride= 2, padding = 1)
        conv2 = nn.Conv2d(hidden_channel, hidden_channel*2, kernel_size= 3, stride= 2, padding = 1)
        conv3 = nn.Conv2d(hidden_channel*2, hidden_channel*2, kernel_size= 3, stride= 2, padding = 1)
        lin4 = nn.Linear(hidden_channel* 2 * 4, 10)
        lin_contra_f = nn.Linear(hidden_channel* 2 * 4, args.contra_dim)
        for lin in [conv1, conv2, conv3, lin4, lin_contra_f]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.final_cls = lin4
        self.contra_head = lin_contra_f
        self._features = nn.Sequential(conv1, nn.ReLU(True), conv2, nn.MaxPool2d(2), nn.ReLU(True), conv3, nn.Flatten())
        if further_cls:
            self.contra_cls = nn.Linear(args.contra_dim, 10)
    def forward(self, input_imgs, need_contra_features = False,need_contra_cls=False):
        features = self._features(input_imgs)
        if need_contra_features:
            features = self.contra_head(features)
            return features
        if need_contra_cls:
            features = self.contra_head(features)
            return self.contra_cls(features)
        out = self.final_cls(features)
        return out
    def get_features_contra(self, input):
        input = input.permute(0,3,1,2)
        features_origin = self._features(input)
        features = self.final_contra_cls_f(features_origin)
        return features







