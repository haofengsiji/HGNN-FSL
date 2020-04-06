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
        if tt.arg.dataset == 'cifar':
            self.last_hidden = self.hidden * 1
        else:
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, layers=[1,1,1,1], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        block=Bottleneck

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock_wrn(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super(BasicBlock_wrn, self).__init__()

        self.drop_rate = drop_rate

        self._preactivate_both = (in_channels != out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preactivate_both:
            x = F.leaky_relu(
                self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            y = F.leaky_relu(
                self.bn1(x),
                inplace=True)  # preactivation only for residual path
            y = self.conv1(y)
        if self.drop_rate > 0:
            y = F.dropout(
                y, p=self.drop_rate, training=self.training, inplace=False)

        y = F.leaky_relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class wrn(nn.Module):
    def __init__(self,emb_size):
        super(wrn, self).__init__()

        input_shape = [1,3,224,224]
        n_classes = emb_size

        base_channels = 16
        widening_factor = 28
        drop_rate = 0.0
        depth = 10

        block = BasicBlock_wrn
        n_blocks_per_stage = (depth - 4) // 6
        assert n_blocks_per_stage * 6 + 4 == depth

        n_channels = [
            base_channels, base_channels * widening_factor,
            base_channels * 2 * widening_factor,
            base_channels * 4 * widening_factor
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks_per_stage,
            block,
            stride=1,
            drop_rate=drop_rate)
        self.stage2 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks_per_stage,
            block,
            stride=2,
            drop_rate=drop_rate)
        self.stage3 = self._make_stage(
            n_channels[2],
            n_channels[3],
            n_blocks_per_stage,
            block,
            stride=2,
            drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    drop_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        drop_rate=drop_rate))
            else:
                stage.add_module(
                    block_name,
                    block(
                        out_channels,
                        out_channels,
                        stride=1,
                        drop_rate=drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx_batch):
        # optimized by Gai
        batch = X.shape[0]
        new_X = torch.zeros(batch, A.shape[1], X.shape[-1]).to(X.device)
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
    def __init__(self,in_dim,hidden = 96, ratio=[2,2,1,1]):
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