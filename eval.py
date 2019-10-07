from torchtools import *
from data import MiniImagenetLoader
from model import EmbeddingImagenet, Unet
import shutil
import os
import random
from train import ModelTrainer

if __name__ == '__main__':
    tt.arg.test_model = 'D-mini_N-5_K-5_B-40_SEED-222' if tt.arg.test_model is None else tt.arg.test_model

    list1 = tt.arg.test_model.split("_")
    param = {}
    for i in range(len(list1)):
        param[list1[i].split("-", 1)[0]] = list1[i].split("-", 1)[1]
    tt.arg.dataset = param['D']
    tt.arg.num_ways = int(param['N'])
    tt.arg.num_shots = int(param['K'])
    tt.arg.meta_batch_size = int(param['B'])

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
    tt.arg.ks = [0.5, 0.5, 0.5, 0.5]

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

    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}'.format(tt.arg.num_ways, tt.arg.num_shots)
    exp_name += '_B-{}'.format(tt.arg.meta_batch_size)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    if not exp_name == tt.arg.test_model:
        print(exp_name)
        print(tt.arg.test_model)
        print('Test model and input arguments are mismatched!')
        AssertionError()

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways)

    test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')

    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                          unet_module=unet_module,
                          data_loader=data_loader)

    checkpoint = torch.load('asset/checkpoints/{}/'.format(exp_name) + 'model_best.pth.tar')
    # checkpoint = torch.load('./trained_models/{}/'.format(exp_name) + 'model_best.pth.tar')

    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    tester.unet_module.load_state_dict(checkpoint['unet_module_state_dict'])
    print("load pre-trained unet done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step)

    tester.eval(partition='test')
