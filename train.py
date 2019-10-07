from torchtools import *
from data import MiniImagenetLoader
from model import EmbeddingImagenet, Unet
import shutil
import os
import random

class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 unet_module,
                 data_loader):
        # set encoder and unet
        self.enc_module = enc_module.to(tt.arg.device)
        self.unet_module = unet_module.to(tt.arg.device)

        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[2, 3], dim=0)
            self.unet_module = nn.DataParallel(self.unet_module, device_ids=[2, 3], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set module parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.unet_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.node_loss = nn.NLLLoss()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_ways * 1
        num_samples = num_supports + num_queries

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways,
                                                                     num_shots=tt.arg.num_shots,
                                                                     seed=iter + tt.arg.seed)

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()  # batch_size x 2 x num_samples x num_samples
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0


            # set as train mode
            self.enc_module.train()
            self.unet_module.train()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x featdim
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            # 2. unet
            node_out = self.unet_module(init_edge,full_data)

            # 3. compute loss
            query_node_out = node_out[:,num_supports:]
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries
            node_loss = [self.node_loss(data.squeeze(1), label.squeeze(1).long()) for (data, label) in
                         zip(query_node_out.chunk(query_node_out.size(1), dim=1), full_label[:, num_supports:].chunk(full_label[:, num_supports:].size(1), dim=1))]
            node_loss = torch.stack(node_loss, dim=0)
            node_loss = torch.mean(node_loss)

            node_loss.backward()

            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            # logging
            tt.log_scalar('train/loss', node_loss, self.global_step)
            tt.log_scalar('train/node_accr', node_accr, self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'unet_module_state_dict': self.unet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            tt.log_step(global_step=self.global_step)


    def eval(self,partition='test', log_flag=True):
        best_acc = 0

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_ways * 1
        num_samples = num_supports + num_queries

        query_node_accrs = []

        # for each iteration
        for iter in range(tt.arg.test_iteration // tt.arg.test_batch_size):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                       num_ways=tt.arg.num_ways,
                                                                       num_shots=tt.arg.num_shots,
                                                                       seed=iter)
            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0

            # set as eval mode
            self.enc_module.eval()
            self.unet_module.eval()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x featdim
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            # 2. unet
            node_out = self.unet_module(init_edge, full_data)

            # node loss,accr
            query_node_out = node_out[:, num_supports:]
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries

            query_node_accrs += [node_accr.item()]

        # logging
        if log_flag:
            tt.log('---------------------------')
            tt.log_scalar('{}/node_accr'.format(partition), np.array(query_node_accrs).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs).mean() * 100,
                    np.array(query_node_accrs).std() * 100,
                    1.96 * np.array(query_node_accrs).std() / np.sqrt(
                        float(len(np.array(query_node_accrs)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs).mean()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        return edge

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}'.format(tt.arg.num_ways, tt.arg.num_shots)
    exp_name += '_B-{}'.format(tt.arg.meta_batch_size)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':

    tt.arg.device = 'cuda:1' if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'dataset'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.meta_batch_size = 40 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1

    # model parameter related
    tt.arg.emb_size = 128
    tt.arg.in_dim = tt.arg.emb_size + tt.arg.num_ways
    tt.arg.ks = [0.5,0.5,0.5,0.5]

    # train, test parameters
    tt.arg.train_iteration = 100000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 1000

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 10000
    tt.arg.dropout = 0.1

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment

    print(set_exp_name())

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    unet_module = Unet(tt.arg.ks,tt.arg.in_dim,tt.arg.num_ways)

    if tt.arg.dataset == 'mini':
        train_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           unet_module=unet_module,
                           data_loader=data_loader)

    trainer.train()



