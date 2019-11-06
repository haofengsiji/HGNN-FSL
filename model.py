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
        # optimized by Gai
        batch = X.shape[0]
        new_X = torch.zeros(batch, A.shape[1], X.shape[-1]).to(tt.arg.device)
        new_X[torch.arange(idx_batch.shape[0]).unsqueeze(-1), idx_batch] = X
        #
        return A, new_X

class GraphPool(nn.Module):

    def __init__(self, k, in_dim, num_classes, num_queries):
        super(GraphPool, self).__init__()
        self.k = k
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.proj = nn.Linear(in_dim, 1).to(tt.arg.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        batch = X.shape[0]
        idx_batch = []
        new_X_batch = []
        new_A_batch = []
        # for each batch
        for i in range(batch):
            num_nodes = A[i, 0].shape[0]
            scores = self.proj(X[i])
            scores = torch.squeeze(scores)
            scores = self.sigmoid(scores/100)

            if tt.arg.pool_mode == 'way':
                num_spports = int((num_nodes - self.num_queries)/self.num_classes)
                idx = []
                values = []
                # pooling by each way
                for j in range(self.num_classes):
                    way_values, way_idx = torch.topk(scores[j*num_spports:(j+1)*num_spports], int(self.k * num_spports))
                    way_idx = way_idx + j*num_spports
                    idx.append(way_idx)
                    values.append(way_values)
                query_values = scores[num_nodes-self.num_queries:]
                query_idx = torch.arange(num_nodes-self.num_queries,num_nodes).long().to(tt.arg.device)
                values = torch.cat(values+[query_values], dim=0)
                idx = torch.cat(idx+[query_idx], dim=0)
            elif tt.arg.pool_mode == 'support':
                num_supports = num_nodes - self.num_queries
                support_values, support_idx = torch.topk(scores[:num_supports], int(self.k * num_supports),largest=True)
                query_values = scores[num_supports:]
                query_idx = torch.arange(num_nodes - self.num_queries, num_nodes).long().to(tt.arg.device)
                values = torch.cat([support_values, query_values], dim=0)
                idx = torch.cat([support_idx, query_idx], dim=0)
            elif tt.arg.pool_mode == 'way&kn':
                num_supports = int((num_nodes - self.num_queries) / self.num_classes)
                idx = []
                values = []
                # pooling by each way
                for j in range(self.num_classes):
                    way_scores = scores[j * num_supports:(j + 1) * num_supports]
                    intra_scores = way_scores - way_scores.mean()
                    _, way_idx = torch.topk(intra_scores,
                                                     int(self.k * num_supports),largest=False)
                    way_values = way_scores[way_idx]
                    way_idx = way_idx + j * num_supports
                    idx.append(way_idx)
                    values.append(way_values)
                query_values = scores[num_nodes - self.num_queries:]
                query_idx = torch.arange(num_nodes - self.num_queries, num_nodes).long().to(tt.arg.device)
                values = torch.cat(values + [query_values], dim=0)
                idx = torch.cat(idx + [query_idx], dim=0)
            elif tt.arg.pool_mode == 'kn':
                num_supports = num_nodes - self.num_queries
                support_scores = scores[:num_supports]
                intra_scores = support_scores - support_scores.mean()
                _, support_idx = torch.topk(intra_scores,
                                        int(self.k * num_supports), largest=False)
                support_values = support_scores[support_idx]
                query_values = scores[num_nodes - self.num_queries:]
                query_idx = torch.arange(num_nodes - self.num_queries, num_nodes).long().to(tt.arg.device)
                values = torch.cat([support_values, query_values], dim=0)
                idx = torch.cat([support_idx, query_idx], dim=0)
            else:
                print('wrong pool_mode setting!!!')
                raise NameError('wrong pool_mode setting!!!')
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
        idx_batch = torch.stack(idx_batch,dim=0).to(tt.arg.device)
        return A, new_X, idx_batch

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
        # parrallel
        x_ij = torch.transpose(x_ij, 1, 3).to(self.conv_last.weight.device)
        #
        A_new = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)

        A_new = F.softmax(A_new,dim=-1)

        return A_new

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim=133,dropout=0.0):
        super(GCN, self).__init__()
        if tt.arg.unet_mode == 'addold':
            self.proj = nn.Linear(in_dim*2, out_dim)
        else:
            self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self,A_new, A_old, X):
        # parrallel
        X = X.to(self.proj.weight.device)
        A_new = A_new.to(X.device)
        A_old = A_old.to(X.device)
        #
        X = self.drop(X)
        if tt.arg.unet_mode == 'addold':
            X1 = torch.bmm(A_new, X)
            X2 = torch.bmm(A_old,X)
            X = torch.cat([X1,X2],dim=-1)
        else:
            X = torch.bmm(A_new, X)
        X = self.proj(X)
        return X

class Unet(nn.Module):

    def __init__(self, ks, in_dim, num_classes, num_queries):
        super(Unet, self).__init__()
        l_n = len(ks)
        self.l_n = l_n
        start_mlp = MLP(in_dim=in_dim)
        start_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
        self.add_module('start_mlp', start_mlp)
        self.add_module('start_gcn', start_gcn)
        for l in range(l_n):
            down_mlp = MLP(in_dim=in_dim)
            down_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
            up_mlp = MLP(in_dim=in_dim)
            up_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
            pool = GraphPool(ks[l],in_dim=in_dim,num_classes=num_classes,num_queries=num_queries)
            unpool = GraphUnpool()

            self.add_module('down_mlp_{}'.format(l),down_mlp)
            self.add_module('down_gcn_{}'.format(l),down_gcn)
            self.add_module('up_mlp_{}'.format(l), up_mlp)
            self.add_module('up_gcn_{}'.format(l), up_gcn)
            self.add_module('pool_{}'.format(l),pool)
            self.add_module('unpool_{}'.format(l),unpool)
        bottom_mlp = MLP(in_dim=in_dim)
        bottom_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
        self.add_module('bottom_mlp', bottom_mlp)
        self.add_module('bottom_gcn', bottom_gcn)

        out_mlp = MLP(in_dim=in_dim*2)
        out_gcn = GCN(in_dim=in_dim*2,out_dim=num_classes)
        self.add_module('out_mlp', out_mlp)
        self.add_module('out_gcn', out_gcn)

    def forward(self, A_init, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        org_X = X
        A_old = A_init
        A_new = self._modules['start_mlp'](X)
        X = self._modules['start_gcn'](A_new, A_old, X)
        for i in range(self.l_n):
            A_old = A_new
            A_new = self._modules['down_mlp_{}'.format(i)](X)
            X = self._modules['down_gcn_{}'.format(i)](A_new, A_old, X)
            adj_ms.append(A_new)
            down_outs.append(X)
            A_new, X, idx_batch = self._modules['pool_{}'.format(i)](A_new, X)
            indices_list.append(idx_batch)
        A_old = A_new
        A_new = self._modules['bottom_mlp'](X)
        X = self._modules['bottom_gcn'](A_new, A_old, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A_old, idx_batch = adj_ms[up_idx], indices_list[up_idx]
            A_old,X = self._modules['unpool_{}'.format(i)](A_old, X, idx_batch)
            X = X.add(down_outs[up_idx])
            A_new = self._modules['up_mlp_{}'.format(up_idx)](X)
            X = self._modules['up_gcn_{}'.format(up_idx)](A_new, A_old, X)
        X = torch.cat([X, org_X], -1)
        A_old = A_new
        A_new = self._modules['out_mlp'](X)
        X = self._modules['out_gcn'](A_new, A_old, X)

        out = F.log_softmax(X,dim=-1)

        return out

class Unet2(nn.Module):

    def __init__(self, ks_1,ks_2,mode_1,mode_2, in_dim, num_classes, num_queries):
        super(Unet2, self).__init__()
        l_n_1 = len(ks_1)
        l_n_2 = len(ks_2)
        l_n = l_n_1 + l_n_2
        self.l_n_1 = l_n_1
        self.l_n_2 = l_n_2
        self.l_n = l_n
        self.mode_1 = mode_1
        self.mode_2 = mode_2
        start_mlp = MLP(in_dim=in_dim)
        start_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
        self.add_module('start_mlp', start_mlp)
        self.add_module('start_gcn', start_gcn)
        for l in range(l_n):
            down_mlp = MLP(in_dim=in_dim)
            down_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
            up_mlp = MLP(in_dim=in_dim)
            up_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
            if l < l_n_1:
                pool = GraphPool(ks_1[l], in_dim=in_dim, num_classes=num_classes, num_queries=num_queries)
            else:
                pool = GraphPool(ks_2[l-l_n_1], in_dim=in_dim, num_classes=num_classes, num_queries=num_queries)
            unpool = GraphUnpool()

            self.add_module('down_mlp_{}'.format(l),down_mlp)
            self.add_module('down_gcn_{}'.format(l),down_gcn)
            self.add_module('up_mlp_{}'.format(l), up_mlp)
            self.add_module('up_gcn_{}'.format(l), up_gcn)
            self.add_module('pool_{}'.format(l),pool)
            self.add_module('unpool_{}'.format(l),unpool)
        bottom_mlp = MLP(in_dim=in_dim)
        bottom_gcn = GCN(in_dim=in_dim,out_dim=in_dim)
        self.add_module('bottom_mlp', bottom_mlp)
        self.add_module('bottom_gcn', bottom_gcn)

        out_mlp = MLP(in_dim=in_dim*2)
        out_gcn = GCN(in_dim=in_dim*2,out_dim=num_classes)
        self.add_module('out_mlp', out_mlp)
        self.add_module('out_gcn', out_gcn)

    def forward(self, A_init, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        A_old = A_init
        A_new = self._modules['start_mlp'](X)
        X = self._modules['start_gcn'](A_new, A_old, X)
        org_X = X
        for i in range(self.l_n):
            if i < self.l_n_1:
                tt.arg.pool_mode = self.mode_1
            else:
                tt.arg.pool_mode = self.mode_2
            A_old = A_new
            A_new = self._modules['down_mlp_{}'.format(i)](X)
            X = self._modules['down_gcn_{}'.format(i)](A_new, A_old, X)
            adj_ms.append(A_new)
            down_outs.append(X)
            A_new, X, idx_batch = self._modules['pool_{}'.format(i)](A_new, X)
            indices_list.append(idx_batch)
        A_old = A_new
        A_new = self._modules['bottom_mlp'](X)
        X = self._modules['bottom_gcn'](A_new, A_old, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A_old, idx_batch = adj_ms[up_idx], indices_list[up_idx]
            A_old,X = self._modules['unpool_{}'.format(i)](A_old, X, idx_batch)
            X = X.add(down_outs[up_idx])
            A_new = self._modules['up_mlp_{}'.format(up_idx)](X)
            X = self._modules['up_gcn_{}'.format(up_idx)](A_new, A_old, X)
        X = torch.cat([X, org_X], -1)
        A_old = A_new
        A_new = self._modules['out_mlp'](X)
        X = self._modules['out_gcn'](A_new, A_old, X)

        out = F.log_softmax(X,dim=-1)

        return out