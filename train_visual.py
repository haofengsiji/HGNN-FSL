from torchtools import *
from data import MiniImagenetLoader
from model_visual import EmbeddingImagenet, Unet, Unet2
import pandas as pd
import shutil
import os
from PIL import Image
import random
from torchvision import transforms

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

            if tt.arg.transductive == True:
                # transduction
                full_node_out = self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]  # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:]  # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)  # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports,
                                                             -1)  # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, -1).unsqueeze(
                    1)  # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],
                                            1)  # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.meta_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)  # (batch_size x num_queries) x (num_support + 1) x num_classes
                node_out = node_out.view(tt.arg.meta_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)  # batch_size x  num_queries x (num_support + 1) x num_classes
                full_node_out = torch.zeros(tt.arg.meta_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            query_node_out = full_node_out[:,num_supports:]
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
            '''
            q0 = query_data[:,0,:].clone()
            q1 = query_data[:,1,:].clone()
            query_data[:, 1, :] = q0
            query_data[:, 0, :] = q1
            ql0 = query_label[:,0].clone()
            ql1 = query_label[:,1].clone()
            query_label[:, 1] = ql0
            query_label[:, 0] = ql1
            '''


            # visual_image
            if tt.arg.visual == None:
                tt.arg.visual = True
            if tt.arg.visual == True and partition=='test':
                tt.arg.iter = iter
                if tt.arg.transductive == True:
                    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
                    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
                    inv_normalize = transforms.Normalize(
                        mean = [-m/s for m,s in zip(mean_pix,std_pix)],
                        std = [1/s for s in std_pix]
                    )
                    if not os.path.exists('visual_%s' % tt.arg.exp_name):
                        os.makedirs('visual_%s' % tt.arg.exp_name)
                    for i in range(tt.arg.test_batch_size):

                        # visual_X(0)
                        if os.path.exists('visual_%s/%03d/feature_record.csv' % (tt.arg.exp_name, tt.arg.iter * tt.arg.test_batch_size + i)):
                            os.remove('visual_%s/%03d/feature_record.csv' % (tt.arg.exp_name, tt.arg.iter * tt.arg.test_batch_size + i))

                        if not os.path.exists('visual_%s/%03d' % (tt.arg.exp_name,iter*tt.arg.test_batch_size+i)):
                            os.makedirs('visual_%s/%03d' % (tt.arg.exp_name,iter*tt.arg.test_batch_size+i))
                        for j in range(support_data.size(1)):
                            pic = inv_normalize(support_data[i, j])
                            im = transforms.ToPILImage()(pic.cpu()).convert('RGB')
                            im.save('visual_%s/%03d/%02d_c%d.jpg' % (tt.arg.exp_name, iter * tt.arg.test_batch_size + i, j, support_label[i, j]))
                        for j in range(query_data.size(1)):
                            pic = inv_normalize(query_data[i, j])
                            im = transforms.ToPILImage()(pic.cpu()).convert('RGB')
                            im.save('visual_%s/%03d/%02d_c%d.jpg' % (tt.arg.exp_name, iter * tt.arg.test_batch_size + i, j+num_supports, query_label[i, j]))

                        # save label
                        np_label = torch.cat((support_label[i, :],query_label[i, :]),dim=0).detach().cpu().numpy()
                        data = [['label'] + list(np_label)]
                        df = pd.DataFrame(data)
                        df.to_csv('visual_%s/%03d/feature_record.csv' % (tt.arg.exp_name, tt.arg.iter * tt.arg.test_batch_size + i),
                                  header=False,
                                  index=False)

                else:
                    support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                          1,1,1)  # batch_size x num_queries x num_support x 3x84x84
                    support_data_tiled = support_data_tiled.reshape(tt.arg.test_batch_size * num_queries, num_supports,
                                                                    3,84,84)  # (batch_size x num_queries) x num_support x 3x84x84
                    query_data_reshaped = query_data.reshape(tt.arg.test_batch_size * num_queries,
                                                             3,84,84).unsqueeze(1)  # (batch_size x num_queries) x 1 x 3x84x84
                    support_label_tiled = support_label.unsqueeze(1).repeat(1, num_queries, 1)  # batch_size x num_queries x num_support
                    support_label_tiled = support_label_tiled.reshape(tt.arg.test_batch_size * num_queries, num_supports)  # (batch_size x num_queries) x num_support
                    query_label_reshaped = query_label.reshape(tt.arg.test_batch_size * num_queries).unsqueeze(1)  # (batch_size x num_queries) x 1

                    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
                    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
                    inv_normalize = transforms.Normalize(
                        mean=[-m / s for m, s in zip(mean_pix, std_pix)],
                        std=[1 / s for s in std_pix]
                    )
                    if not os.path.exists('visual_%s' % tt.arg.exp_name):
                        os.makedirs('visual_%s' % tt.arg.exp_name)
                    trand_batch_size = tt.arg.test_batch_size*num_queries
                    for i in range(trand_batch_size):

                        # visual_X(0)
                        if os.path.exists('visual_%s/%03d/feature_record.csv' % (
                                            tt.arg.exp_name, tt.arg.iter * trand_batch_size + i)):
                            os.remove('visual_%s/%03d/feature_record.csv' % (
                                        tt.arg.exp_name, tt.arg.iter * trand_batch_size + i))

                        if not os.path.exists('visual_%s/%03d' % (tt.arg.exp_name, iter * trand_batch_size + i)):
                            os.makedirs('visual_%s/%03d' % (tt.arg.exp_name, iter * trand_batch_size + i))
                        for j in range(support_data_tiled.size(1)):
                            pic = inv_normalize(support_data_tiled[i, j])
                            im = transforms.ToPILImage()(pic.cpu()).convert('RGB')
                            im.save('visual_%s/%03d/%02d_c%d.jpg'
                                    % (tt.arg.exp_name, iter * trand_batch_size + i, j, support_label_tiled[i, j]))
                        for j in range(query_data_reshaped.size(1)):
                            pic = inv_normalize(query_data_reshaped[i, j])
                            im = transforms.ToPILImage()(pic.cpu()).convert('RGB')
                            im.save('visual_%s/%03d/%02d_c%d.jpg'
                                    % (tt.arg.exp_name, iter * trand_batch_size + i, j + num_supports, query_label_reshaped[i, j]))

                        # save label
                        np_label = torch.cat((support_label_tiled[i, :], query_label_reshaped[i, :]), dim=0).detach().cpu().numpy()
                        data = [['label'] + list(np_label)]
                        df = pd.DataFrame(data)
                        df.to_csv('visual_%s/%03d/feature_record.csv' % (
                        tt.arg.exp_name, tt.arg.iter * trand_batch_size + i),
                                  header=False,
                                  index=False)




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

            if tt.arg.transductive == True:
                # transduction
                full_node_out = self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]  # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:]  # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)  # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.test_batch_size * num_queries, num_supports,
                                                             -1)  # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.test_batch_size * num_queries, -1).unsqueeze(
                    1)  # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],
                                            1)  # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.test_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)  # (batch_size x num_queries) x (num_support + 1) x num_classes
                node_out = node_out.view(tt.arg.test_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)  # batch_size x  num_queries x (num_support + 1) x num_classes
                full_node_out = torch.zeros(tt.arg.test_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            query_node_out = full_node_out[:, num_supports:]
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries

            query_node_accrs += [node_accr.item()]

            # visual parameters refesh
            tt.arg.pool_count = 0
            if tt.arg.iter > 0:
                tt.arg.visual = False

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
    exp_name += '_B-{}_T-{}'.format(tt.arg.meta_batch_size,tt.arg.transductive)
    exp_name += '_P-{}_Pj-{}'.format(tt.arg.pool_mode,tt.arg.proj_mode)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':

    tt.arg.device = 'cuda:1' if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'dataset'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_queries = tt.arg.num_ways*1
    tt.arg.num_supports = tt.arg.num_ways*tt.arg.num_shots
    tt.arg.transductive = True if tt.arg.transductive is None else tt.arg.transductive
    if tt.arg.transductive == False:
        tt.arg.meta_batch_size = 20
    else:
        tt.arg.meta_batch_size = 40
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1

    # model parameter related
    tt.arg.emb_size = 128
    tt.arg.in_dim = tt.arg.emb_size + tt.arg.num_ways

    tt.arg.pool_mode = 'support' # 'com'/'way'/'support'/'inter&intra'/'way&inter'
    tt.arg.proj_mode = 'multiHead'# 'com'/'multiHead'
    unet2_flag = False # the label of using unet2

    # confirm ks
    if tt.arg.num_shots == 1 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'com':  # 'com': pooling all;
            tt.arg.ks = [0.7, 0.5]  # 6->4>2
        elif tt.arg.pool_mode == 'support': # 'support': pooling on support
            tt.arg.ks = [0.6,0.5] # 5->3->1
        elif tt.arg.pool_mode == 'way&inter': # pooling on support by inter way
            tt.arg.ks = [0.6,0.5] # 5->3->1
        else:
            print('wrong mode setting!!!')
            AssertionError()
    elif tt.arg.num_shots == 5 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'com':  # 'com' pooling all; no 'way&query'
            tt.arg.ks = [0.62, 0.375, 0.67, 0.5]  # 26->16->6->4->2
        elif tt.arg.pool_mode == 'support':
            tt.arg.ks = [0.6, 0.34, 0.6, 0.5]  # 25->15->5->3->1
        elif tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5] # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'inter&intra':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&intra'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'way&inter'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            AssertionError()
    elif tt.arg.num_shots == 1 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'com':  # 'com': pooling all;
            tt.arg.ks = [0.8, 0.75]  # 10->8->6
        elif tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'way&inter':  # pooling on support by inter way
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        else:
            print('wrong mode setting!!!')
            AssertionError()
    elif tt.arg.num_shots == 5 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'com':  # 'com' pooling all; no 'way&query'
            tt.arg.ks = [0.67, 0.5, 0.8, 0.75]  # 30->20->10->8->6
        elif tt.arg.pool_mode == 'support':
            tt.arg.ks = [0.6, 0.34, 0.6, 0.5]  # 25->15->5->3->1
        elif tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'inter&intra':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&intra'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'way&inter'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            AssertionError()
    else:
        print('wrong shot and T settings!!!')
        AssertionError()


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

    if tt.arg.transductive == False:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, 1)
        else:
            unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, 1)
    else:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)
        else:
            unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)


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



