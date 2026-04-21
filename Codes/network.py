import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl, math
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
from tools.ema import EMA

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

print(grid_h)
print(grid_w)

# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2

#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug


# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # network
    if is_training == True:
        # aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        # H_motion, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
        H_motion, mesh_motion, oe1, oe2 = net(input1_tensor, input2_tensor)
    else:
        H_motion, mesh_motion, oe1, oe2 = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
        # 单个仿射变换矩阵，将坐标映射到图像中心
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1) # 扩展到整个批次
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)
        # 将H中的像素坐标变换映射到图像中心
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (img_h, img_w))

    H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
    output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat, (img_h, img_w))

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion


    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]

    # calculate the overlapping regions to apply shape-preserving constraints
    overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)


    out_dict = {}
    out_dict.update(output_H=output_H, output_H_inv = output_H_inv, warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, mesh1 = rigid_mesh, mesh2 = mesh, overlap = overlap, edge1=oe1, edge2=oe2)


    return out_dict


# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion, mesh_motion, oe1, oe2 = net(resized_input1, resized_input2)

    H_motion = H_motion.reshape(-1, 4, 2)
    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    #print(out_width)
    #print(out_height)

    # get warped img1
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)

    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor+1, mask), 1), I_mat, (out_height.int(), out_width.int()))

    torch.cuda.empty_cache()
    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))


    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...]-1, final_warp1_mask = homo_output[:, 3:6, ...], final_warp2=tps_output[:, 0:3, ...]-1, final_warp2_mask = tps_output[:, 3:6, ...], mesh1=rigid_mesh, mesh2=mesh_trans)

    return out_dict



# define and forward
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=16384, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            EMA(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=32768, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )

        self.eam = EGM_each_scpc_scaleadd()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """backbone"""
        ssl._create_default_https_context = ssl._create_unverified_context
        resnet50_model = models.resnet.resnet50(pretrained=True)
        vgg_model = models.vgg.vgg16()

        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
            vgg_model = vgg_model.cuda()
        
        #self.extractor_2, self.extractor_4, self.extractor_8, self.extractor_16, self.extractor_32 = self.get_vgg16_FeatureMap(vgg_model)
        self.extractor_2, self.extractor_4, self.extractor_8, self.extractor_16, self.extractor_32 = self.get_res50_FeatureMap(resnet50_model)
        
    def get_vgg16_FeatureMap(self, vgg16_model):
        features = vgg16_model.features
        extract2 = nn.Sequential(*list(features.children())[:4])
        extract4 = nn.Sequential(*list(features.children())[4:9])
        extract8 = nn.Sequential(*list(features.children())[9:16])
        extract16 = nn.Sequential(*list(features.children())[16:23])
        extract32 = nn.Sequential(*list(features.children())[23:30])
        
        return extract2, extract4, extract8, extract16, extract32


    def get_res50_FeatureMap(self, resnet50_model):

        extract2 = nn.Sequential(resnet50_model.conv1)
        extract4 = nn.Sequential(resnet50_model.bn1,
                                 resnet50_model.relu,
                                 resnet50_model.maxpool,
                                 resnet50_model.layer1,
                                 )
        extract8 = nn.Sequential(resnet50_model.layer2)
        extract16 = nn.Sequential(resnet50_model.layer3)
        extract32 = nn.Sequential(resnet50_model.layer4)

        return extract2, extract4, extract8, extract16, extract32

    # forward
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()
        '''
        '''
        f1_2,f2_2 = self.extractor_2(input1_tesnor), self.extractor_2(input2_tesnor)
        f1_4, f2_4 = self.extractor_4(f1_2), self.extractor_4(f2_2)
        f1_8, f2_8 = self.extractor_8(f1_4), self.extractor_8(f2_4)
        f1_16, f2_16 = self.extractor_16(f1_8), self.extractor_16(f2_8)
        f1_32, f2_32 = self.extractor_32(f1_16), self.extractor_32(f2_16)
        print("f1_2.shape", f1_2.shape)
        print("f1_4.shape", f1_4.shape)
        print("f1_8.shape", f1_8.shape)
        print("f1_16.shape", f1_16.shape)
        print("f1_32.shape", f1_32.shape)
        '''
        '''
        edge1, edge2 = self.eam(f1_32, f1_16, f1_8, f1_4, f1_2), self.eam(f2_32, f2_16, f2_8, f2_4, f2_2)
        edge_att1, edge_att2 = torch.sigmoid(edge1), torch.sigmoid(edge2)
        oe1 = F.interpolate(edge_att1, scale_factor=2, mode='bilinear', align_corners=False)
        oe2 = F.interpolate(edge_att2, scale_factor=2, mode='bilinear', align_corners=False)

        ######### stage 1
        correlation_32 = self.CCL(f1_16, f2_16)
        print("correlation_32", correlation_32.shape)
        temp_1 = self.regressNet1_part1(correlation_32)
        print("temp_1", temp_1.shape)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        print("temp_1", temp_1.shape)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)


        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        warp_feature_2_64 = torch_homo_transform.transformer(f2_8, H_mat, (int(img_h/8), int(img_w/8)))

        ######### stage 2
        correlation_64 = self.CCL(f1_8, warp_feature_2_64)
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)

        print(offset_2.shape)

        return offset_1, offset_2, oe1, oe2


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow

# multimask
class ReceptiveConv(nn.Module):
    def __init__(self, inplanes=64, planes=64, dilation=[1,2,4,8], baseWidth=16, scale=4, aggregation=True, use_dwconv=False):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        #self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            if use_dwconv:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], groups=self.width, bias=False))
            else:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.aggregation = aggregation

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        spx = torch.split(x, self.width, 1)
        out = []
        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i==0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out
    
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class EGM_each_scpc_scaleadd(nn.Module):
    def __init__(self,scale=None):
        super(EGM_each_scpc_scaleadd, self).__init__()
        if scale == None:
            scale = nn.Parameter(torch.ones(5))
        self.scale = scale
        '''
        xiugaichicun
        '''
        self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(128, 64, 1)
        self.reduce3 = nn.Conv2d(256, 128, 1)
        self.reduce4 = nn.Conv2d(512, 256, 1)
       
        self.reduce5 = nn.Conv2d(512, 512, 1)
      


        self.scpc1 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc2 = ReceptiveConv(128, 64, [1,2,4,8])
        self.scpc3 = ReceptiveConv(256, 128, [1,2,4,8])
        self.scpc4 = ReceptiveConv(512, 256, [1,2,3,4])
        self.scpc5 = ReceptiveConv(512, 512, [1,2,3,4])
        
        self.reduce54 = nn.Conv2d(512, 256, 1)
        self.reduce43 = nn.Conv2d(256, 128, 1)
        self.reduce32 = nn.Conv2d(128, 64, 1)
        self.conv12 = nn.Conv2d(64, 64, 1)
        '''
        '''
        
        self.conv = ConvBNR(64, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):#!!!!!!!
        size1=x1.size()[2:]
        size2=x2.size()[2:]
        size3=x3.size()[2:]
        size4=x4.size()[2:]
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x5 = self.reduce54(self.scpc5(self.reduce5(x5)))
        x5 = F.interpolate(x5, size4, mode='bilinear', align_corners=False)
        
        x4 = self.scpc4(self.reduce4(x4))
        x4 = self.reduce43(x4+self.scale[4]*x5)
        
        x4 = F.interpolate(x4, size3, mode='bilinear', align_corners=False)
        
        x3 = self.scpc3(self.reduce3(x3))
        x3 = self.reduce32(x3+self.scale[3]*x4)
        
        x3 = F.interpolate(x3, size2, mode='bilinear', align_corners=False)
        
        x2 = self.scpc2(self.reduce2(x2))
        x2 = self.conv12(x2+self.scale[2]*x3)
        
        x2 = F.interpolate(x2, size1, mode='bilinear', align_corners=False)
        
        #x1 = self.scpc1(self.reduce1(x1))
        x1 = self.scpc1(x1)
        x1 = self.conv(self.scale[0]*x1+self.scale[1]*x2)
        
        out = self.edge(x1)
        
        return out
    
