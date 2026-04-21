import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
#import random
#import numpy as np

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
"""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
"""
def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(args):
    # path to save the model files
    MODEL_DIR = os.path.join(last_path, args.model_dir)
    # create folders if it dose not exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
    """
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    """


    # define the network
    net = Network()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    
    net.to(device=device)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        
        # 1. 获取当前网络的状态字典
        model_dict = net.state_dict()
        start_epoch = 0
        glob_iter = 0
        # 2. 筛选只需要加载的预训练参数(特征提取和边缘提取部分)
        pretrained_dict = {
            k: v for k, v in checkpoint['model'].items() 
            if 'extractor_' in k or 'eam' in k
        }
        
        # 3. 更新当前网络字典(只更新特征提取部分)
        model_dict.update(pretrained_dict)
        
        # 4. 加载筛选后的参数
        net.load_state_dict(model_dict, strict=False)
        
        # 5. 冻结特征提取和边缘提取网络
        for name, param in net.named_parameters():
            if 'extractor_' in name or 'eam' in name:
                param.requires_grad = False

        print(f'Loaded feature extractor and edge network from {model_path}')
        print('Frozen feature extractor and edge network parameters')
    else:
        start_epoch = 0
        glob_iter = 0
        print('Training from scratch!')

    # 定义优化器(只优化回归网络部分)
    regress_params = [
        {'params': net.module.regressNet1_part1.parameters()},
        {'params': net.module.regressNet1_part2.parameters()},
        {'params': net.module.regressNet2_part1.parameters()},
        {'params': net.module.regressNet2_part2.parameters()}
    ] if torch.cuda.device_count() > 1 else [
        {'params': net.regressNet1_part1.parameters()},
        {'params': net.regressNet1_part2.parameters()},
        {'params': net.regressNet2_part1.parameters()},
        {'params': net.regressNet2_part2.parameters()}
    ]

    optimizer = optim.Adam(regress_params, lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 验证参数冻结情况
    print("\nTrainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    print("##################start training#######################")
    score_print_fre = 100
    min_loss = float('inf')
    min_loss_state = None

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma , loss_edge_sigma = 0., 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()
            edge1 = batch_value[2].float()
            edge2 = batch_value[3].float()
            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()
                edge1, edge2 = edge1.cuda(), edge2.cuda() 

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
            # result
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']
            edge1_p, edge2_p = batch_out['edge1'], batch_out['edge2']
            # calculate loss for overlapping regions
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            # calculate loss for non-overlapping regions
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)

            #calculate loss for edge
            loss_edge = dice_loss(edge1_p, edge1) + dice_loss(edge2_p, edge2)
            total_loss = overlap_loss + nonoverlap_loss + loss_edge
            total_loss.backward()
            current_loss = total_loss

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']],
                max_norm=3,
                norm_type=2
            )
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_edge_sigma += loss_edge.item()
            loss_sigma += total_loss.item()

            # print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                average_loss_edge = loss_edge_sigma / score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma , loss_edge_sigma = 0. , 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} edge loss{:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, average_loss_edge , optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                # writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                # writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                # writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                # writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)
                writer.add_scalar('edge loss', average_loss_edge, glob_iter)

            glob_iter += 1

        scheduler.step()
        # save model
        if current_loss < min_loss and min_loss is not None:
            min_loss = current_loss
            min_loss_state = {
                'model': net.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                "glob_iter": glob_iter,
                'frozen_params': [name for name, param in net.named_parameters() if not param.requires_grad]  # 记录冻结参数
            }

        # 替换原来的保存代码
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            
            # 强制保存所有参数（包括冻结的）
            state = {
                'model': net.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                "glob_iter": glob_iter,
                'frozen_params': [name for name, param in net.named_parameters() if not param.requires_grad]  # 记录冻结参数
            }
            torch.save(state, model_save_path)

    if min_loss_state is not None:
        min_loss_filename = f'epoch_loss={min_loss:.4f}.pth'
        min_loss_model_path = os.path.join(MODEL_DIR, min_loss_filename)
        torch.save(min_loss_state, min_loss_model_path)
    print("##################end training#######################")

if __name__=="__main__":

    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='./ship_data/train/')
    parser.add_argument('--model_dir', type=str, default='model_train_ship/scpc/')
    #parser.add_argument('--seed', type=int, default=42)

    # parse the arguments
    args = parser.parse_args()
    print(args)

    #set_seed(args.seed)

    # train
    train(args)
