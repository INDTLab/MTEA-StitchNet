# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_output_model, Network
from dataset import *
import os
import cv2

from piqe import piqe
import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


def draw_mesh_on_warp(warp, f_local):


    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return




def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    MODEL_DIR = os.path.join(last_path, args.model_dir)

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    #nl: set num_workers = the number of cpus
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        
    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        #model_path = '/opt/data/private/nl/Repository/Unsupervised_Mesh_Stitching/UDISv2-88/UDISv2-Homo_TPS88-10grid_NO-res50-new3/model/epoch150_model.pth'
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    # create folders if it dose not exist

    #定义保存路径
    # path_ave_fusion = args.test_path + args.out_dir +'/ave_fusion/'
    # if not os.path.exists(path_ave_fusion):
    #     os.makedirs(path_ave_fusion)
    path_warp1 = args.test_path + args.out_dir + '/warp1/'
    if not os.path.exists(path_warp1):
        os.makedirs(path_warp1)
    path_warp2 = args.test_path + args.out_dir + '/warp2/'
    if not os.path.exists(path_warp2):
        os.makedirs(path_warp2)
    path_mask1 = args.test_path + args.out_dir + '/mask1/'
    if not os.path.exists(path_mask1):
        os.makedirs(path_mask1)
    path_mask2 = args.test_path + args.out_dir + '/mask2/'
    if not os.path.exists(path_mask2):
        os.makedirs(path_mask2)
    path_edge1 = args.test_path + args.out_dir + '/edge1/'
    if not os.path.exists(path_edge1):
        os.makedirs(path_edge1)
    path_edge2 = args.test_path + args.out_dir + '/edge2/'
    if not os.path.exists(path_edge2):
        os.makedirs(path_edge2)
    path_o_edge1 = args.test_path + args.out_dir + '/o_edge1/'
    if not os.path.exists(path_o_edge1):
        os.makedirs(path_o_edge1)
    path_o_edge2 = args.test_path + args.out_dir + '/o_edge2/'
    if not os.path.exists(path_o_edge2):
        os.makedirs(path_o_edge2)

    piqe_inpu1_list = []
    piqe_inpu2_list = []
    piqe_warp_list = []

    net.eval()
    for i, batch_value in enumerate(test_loader):

        #if i != 975:
        #    continue

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
        edge1 = batch_value[2].float()
        edge2 = batch_value[3].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_output_model(net, inpu1_tesnor, inpu2_tesnor)

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']
        """
        oe1 = batch_out['edge1']
        oe2 = batch_out['edge2']
        """
        gt_edge1 = ((edge1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        gt_edge2 = ((edge2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        #final_edge1 = ((oe1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        #final_edge2 = ((oe2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        final_mesh2 = final_mesh2[0].cpu().detach().numpy()
        final_warp1 = draw_mesh_on_warp(final_warp1, final_mesh1)
        final_warp2 = draw_mesh_on_warp(final_warp2, final_mesh2)
        """
        inpu1 = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        inpu2 = ((inpu2_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        """

        #输出图像保存设置
        path = path_edge1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, gt_edge1)
        path = path_edge2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, gt_edge2)
        """
        path = path_o_edge1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_edge1)
        path = path_o_edge2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_edge2)
        """
        print('i = {}'.format(i+1))
        path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1)
        path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2)
        path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1_mask*255)
        path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2_mask*255)

        # ave_fusion = cv2.addWeighted(final_warp1, 0.5, final_warp2, 0.5, 0)
        # path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, ave_fusion)

        """
        #打分
        piqe_inpu1_score, _, _, _ = piqe(inpu1)
        piqe_inpu2_score, _, _, _ = piqe(inpu2)
        piqe_warp_score, _, _, _ = piqe(ave_fusion)
        piqe_inpu1_list.append(piqe_inpu1_score)
        piqe_inpu2_list.append(piqe_inpu2_score)
        piqe_warp_list.append(piqe_warp_score)
        print('i = {}, piqe_inpu1 = {:.6f}, piqe_inpu2 = {:.6f}, piqe_warp = {:.6f}'.format( i+1, piqe_inpu1_score, piqe_inpu2_score, piqe_warp_score))
        print('i = {}, piqe_warp = {:.6f}'.format( i+1, piqe_warp_score))
        """
        torch.cuda.empty_cache()
    """
    print("=================== Analysis ==================")
    print("piqe")
    piqe_in = [(x + y) / 2 for x, y in zip(piqe_inpu1_list, piqe_inpu2_list)]#将两个列表取平均值，生成新列表piqe_in
    piqe_in.sort(reverse = False)#升序排序
    #对输入分段打分
    piqe_in_30 = piqe_in[0 : 331]
    piqe_in_60 = piqe_in[331: 663]
    piqe_in_100 = piqe_in[663: -1]
    print("top 30%", np.mean(piqe_in_30))
    print("top 30~60%", np.mean(piqe_in_60))
    print("top 60~100%", np.mean(piqe_in_100))
    print('input average piqe:', np.mean(piqe_in))
    #对配准图像进行打分
    
    piqe_warp_list.sort(reverse = False)
    piqe_warp_30 = piqe_warp_list[0 : 331]
    piqe_warp_60 = piqe_warp_list[331: 663]
    piqe_warp_100 = piqe_warp_list[663: -1]
    print("top 30%", np.mean(piqe_warp_30))
    print("top 30~60%", np.mean(piqe_warp_60))
    print("top 60~100%", np.mean(piqe_warp_100))
    print('warp average piqe:', np.mean(piqe_warp_list))
    print("##################end testing#######################")
    """

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # /opt/data/private/nl/Data/UDIS-D/testing/  or  /opt/data/private/nl/Data/UDIS-D/training/
    parser.add_argument('--test_path', type=str, default='/data2/houzhe/stitch/video-ship/test_11class/lh101')
    parser.add_argument('--out_dir', type=str, default='/scpc')
    parser.add_argument('--model_dir', type=str, default='ship3700/scpc', help="./model_orgin")


    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
