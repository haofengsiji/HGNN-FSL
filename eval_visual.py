from torchtools import *
from data import MiniImagenetLoader
from model_visual import EmbeddingImagenet, Unet,Unet2
import shutil
import os
import random
from train_visual import ModelTrainer

if __name__ == '__main__':
    tt.arg.test_model = 'D-mini_N-5_K-5_B-40_T-True_P-way&inter_Pj-multiHead_SEED-222' if tt.arg.test_model is None else tt.arg.test_model

    list1 = tt.arg.test_model.split("_")
    param = {}
    for i in range(len(list1)):
        param[list1[i].split("-", 1)[0]] = list1[i].split("-", 1)[1]
    tt.arg.dataset = param['D']
    tt.arg.num_ways = int(param['N'])
    tt.arg.num_shots = int(param['K'])
    tt.arg.meta_batch_size = int(param['B'])
    tt.arg.transductive = False if param['T'] == 'False' else True
    tt.arg.pool_mode = param['P']
    tt.arg.proj_mode = param['Pj']


    tt.arg.device = 'cuda:1' if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'dataset'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_queries = tt.arg.num_ways * 1
    tt.arg.num_supports = tt.arg.num_ways * tt.arg.num_shots
    tt.arg.transductive = True
    if tt.arg.transductive == False:
        tt.arg.meta_batch_size = 20
    else:
        tt.arg.meta_batch_size = 40
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1

    # model parameter related
    tt.arg.emb_size = 128
    tt.arg.in_dim = tt.arg.emb_size + tt.arg.num_ways

    tt.arg.pool_mode = 'intra'  # 'com'/'way'/'support'/'inter&intra'/'way&inter'/'intra'
    tt.arg.proj_mode = 'multiHead'  # 'com'/'multiHead'
    unet2_flag = False  # the label of using unet2

    # confirm ks
    if tt.arg.num_shots == 1 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'com':  # 'com': pooling all;
            tt.arg.ks = [0.7, 0.5]  # 6->4>2
        elif tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'way&inter':  # pooling on support by inter way
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'intra': # left close support node
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
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'inter&intra':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&intra'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'way&inter'
            unet2_flag = True
        elif tt.arg.pool_mode == 'intra':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&intra'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'intra'
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
        elif tt.arg.pool_mode == 'intra': # left close support node
            tt.arg.ks = [0.6,0.5] # 5->3->1
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
        elif tt.arg.pool_mode == 'intra':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&intra'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'intra'
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

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.exp_name = 'D-{}'.format(tt.arg.dataset)
    tt.arg.exp_name += '_N-{}_K-{}'.format(tt.arg.num_ways, tt.arg.num_shots)
    tt.arg.exp_name += '_B-{}_T-{}'.format(tt.arg.meta_batch_size, tt.arg.transductive)
    tt.arg.exp_name += '_P-{}_Pj-{}'.format(tt.arg.pool_mode, tt.arg.proj_mode)
    tt.arg.exp_name += '_SEED-{}'.format(tt.arg.seed)

    if not tt.arg.exp_name == tt.arg.test_model:
        print(tt.arg.exp_name)
        print(tt.arg.test_model)
        print('Test model and input arguments are mismatched!')
        AssertionError()

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
            unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways,
                                tt.arg.num_queries)

    test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')

    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                          unet_module=unet_module,
                          data_loader=data_loader)

    checkpoint = torch.load('asset/checkpoints/{}/'.format(tt.arg.exp_name) + 'model_best.pth.tar',map_location=tt.arg.device)
    # checkpoint = torch.load('./trained_models/{}/'.format(tt.arg.exp_name) + 'model_best.pth.tar',map_location=tt.arg.device)

    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    tester.unet_module.load_state_dict(checkpoint['unet_module_state_dict'])
    print("load pre-trained unet done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step)

    tester.eval(partition='test')
