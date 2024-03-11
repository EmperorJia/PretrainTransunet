import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse, inference_slice, inference_batch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='nii', help='experiment_name')
parser.add_argument('--images_dir', type=str,
                    default='/home/peijia/medical_dataset/image', help='image dir')
parser.add_argument('--mask_dir', type=str,
                    default='/home/peijia/medical_dataset/mask', help='mask dir')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
# 【重要通知，这里设定预处理后的图像大小】
parser.add_argument('--img_size', type=int,
                    default=160, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

# def inference(args, model, test_save_path=None):
#     from datasets.dataset_synapse import TetsNiiDataset
#     split_path = '/home/peijia/medical_dataset/'
#     split_list = ['test_img.txt']
#     for i, file_name in enumerate(split_list):
#         split_list[i] = split_path + file_name
    
#     db_test = TetsNiiDataset(None, args.images_dir, args.mask_dir, split_list, split="test")
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label = sampled_batch["image"], sampled_batch["label"]
#         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=None)
#         metric_list += np.array(metric_i)
#         logging.info('idx %s mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"




if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'nii': {
            'root_path': '/home/peijia/medical_dataset',
            'list_dir': '/home/peijia/medical_dataset',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    # args.exp = 'No_1_' + dataset_name + str(args.img_size)
    # snapshot_path = "./log_{}".format(args.exp)
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_'+str(args.img_size)
    # snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path'
    args.exp = 'trained_model/pretrain_complete'
    snapshot_path = 'trained_model/pretrain_complete'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # trainer_synapse(args, net, snapshot_path)

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './' + snapshot_path + '/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    
    test_save_path = None
    # inference(ar  gs, net, test_save_path)

    split_path='/home/peijia/medical_dataset/'
    split_list=['test_img.txt']
    from datasets.dataset_synapse import NiiDataset, pretrain_dataset
    for i, file_name in enumerate(split_list):
        split_list[i] = split_path + file_name
    db_test = pretrain_dataset(None, args.images_dir, args.mask_dir, split_list, split="test")
    
    inference_batch(args, net, db_test, test_save_path)
