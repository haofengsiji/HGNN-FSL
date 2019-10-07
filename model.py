from torchtools import *
from collections import OrderedDict
import math

# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))

class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx_batch):
        batch = X.shape[0]
        new_X_batch = []
        for i in range(batch):
            new_X = torch.zeros([A[i].shape[0], X[i].shape[1]]).to(tt.arg.device)
            new_X[idx_batch[i]] = X[i]
            new_X_batch.append(new_X)
        new_X = torch.stack(new_X_batch,dim=0).to(tt.arg.device)
        return A, new_X

class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1).to(tt.arg.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        batch = X.shape[0]
        idx_batch = []
        new_X_batch = []
        new_A_batch = []
        # for each batch
        for i in range(batch):
            scores = self.proj(X[i])
            scores = torch.squeeze(scores)
            scores = self.sigmoid(scores/100)
            num_nodes = A[i,0].shape[0]
            values, idx = torch.topk(scores, int(self.k*num_nodes))
            new_X = X[i,idx, :]
            values = torch.unsqueeze(values, -1)
            new_X = torch.mul(new_X, values)
            new_A = A[i,idx, :]
            new_A = new_A[:, idx]
            idx_batch.append(idx)
            new_X_batch.append(new_X)
            new_A_batch.append(new_A)
        A = torch.stack(new_A_batch,dim=0).to(tt.arg.device)
        new_X = torch.stack(new_X_batch,dim=0).to(tt.arg.device)
        return A, new_X, idx_batch

class Unet(nn.Module):

    def __init__(self, ks, in_dim, num_classes):
        super(Unet, self).__init__()
        l_n = len(ks)
        self.l_n = l_n
        start_mlp = MLP(in_dim=in_dim)
        start_gcn = GCN(in_dim=in_dim*2,out_dim=in_dim)
        self.add_module('start_mlp', start_mlp)
        self.add_module('start_gcn', start_gcn)
        for l in range(l_n):
            down_mlp = MLP(in_dim=in_dim)
            down_gcn = GCN(in_dim=in_dim*2,out_dim=in_dim)
            up_mlp = MLP(in_dim=in_dim)
            up_gcn = GCN(in_dim=in_dim*2,out_dim=in_dim)
            pool = GraphPool(ks[l],in_dim=in_dim)
            unpool = GraphUnpool()

            self.add_module('down_mlp_{}'.format(l),down_mlp)
            self.add_module('down_gcn_{}'.format(l),down_gcn)
            self.add_module('up_mlp_{}'.format(l), up_mlp)
            self.add_module('up_gcn_{}'.format(l), up_gcn)
            self.add_module('pool_{}'.format(l),pool)
            self.add_module('unpool_{}'.format(l),unpool)
        bottom_mlp = MLP(in_dim=in_dim)
        bottom_gcn = GCN(in_dim=in_dim*2,out_dim=in_dim)
        self.add_module('bottom_mlp', bottom_mlp)
        self.add_module('bottom_gcn', bottom_gcn)

        out_mlp = MLP(in_dim=in_dim*2)
        out_gcn = GCN(in_dim=in_dim*2*2,out_dim=num_classes)
        self.add_module('out_mlp', out_mlp)
        self.add_module('out_gcn', out_gcn)

    def forward(self, A_init, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        A_new = self._modules['start_mlp'](X)
        X = self._modules['start_gcn'](A_new, A_init, X)
        org_X = X
        for i in range(self.l_n):
            A_new = self._modules['down_mlp_{}'.format(i)](X)
            X = self._modules['down_gcn_{}'.format(i)](A_new, A_init, X)
            adj_ms.append(A_init)
            down_outs.append(X)
            A_init, X, idx_batch = self._modules['pool_{}'.format(i)](A_init, X)
            indices_list.append(idx_batch)
        A_new = self._modules['bottom_mlp'](X)
        X = self._modules['bottom_gcn'](A_new, A_init, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A_init, idx_batch = adj_ms[up_idx], indices_list[up_idx]
            A_init,X = self._modules['unpool_{}'.format(i)](A_init, X, idx_batch)
            A_new = self._modules['up_mlp_{}'.format(i)](X)
            X = self._modules['up_gcn_{}'.format(i)](A_new, A_init, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], -1)
        A_new = self._modules['out_mlp'](X)
        X = self._modules['out_gcn'](A_new, A_init, X)

        out = F.log_softmax(X,dim=-1)

        return out

class MLP(nn.Module):
    def __init__(self,in_dim,hidden=96,ratio=[2,2,1,1]):
        super(MLP, self).__init__()
        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_dim,
                                              out_channels=hidden*ratio[0],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[0]),
                                    nn.LeakyReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden*ratio[0],
                                              out_channels=hidden*ratio[1],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[1]),
                                    nn.LeakyReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[1],
                                              out_channels=hidden * ratio[2],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[2]),
                                    nn.LeakyReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[2],
                                              out_channels=hidden * ratio[3],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[3]),
                                    nn.LeakyReLU())
        self.conv_last = nn.Conv2d(in_channels=hidden * ratio[3],
                                              out_channels=1,
                                              kernel_size=1)
    def forward(self,X):
        # compute abs(x_i, x_j)
        x_i = X.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        A_new = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)

        A_new = F.softmax(A_new,dim=-1)

        return A_new

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim=133,dropout=0.0):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self,A_new, A_init, X):
        X = self.drop(X)
        X1 = torch.bmm(A_new, X)
        X2 = torch.bmm(A_init,X)
        X = torch.cat([X1,X2],dim=-1)
        X = self.proj(X)
        return X


