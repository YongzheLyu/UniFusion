#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from codecs import strict_errors
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
import torch.nn.functional as F
from scene.regulation import compute_plane_smoothness
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, opt):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        #新增deformation network
        self._deformation = deform_network(opt)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.use_mip_filter = False
        #self.opt = opt
        # 添加deformation相关变量
        self._deformation_table = torch.empty(0)
        self._deformation_accum = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            self._deformation_accum,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        deform_state,
        self._deformation_table,
        self._deformation_accum,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling)
        if self.use_mip_filter:
            scales = torch.square(scales) + torch.square(self.mip_filter)
            scales = torch.sqrt(scales)
        #print(scales.min(), scales.max())
        return scales #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        
        return self._xyz
    
    def get_deformed_xyz(self, time):
        # 调用形变网络计算偏移，返回随时间变形后的坐标
        # 适配2DGS的形变计算
        if len(time.shape) == 0:  # 单个时间值
            time = time.unsqueeze(0).expand(self._xyz.shape[0], 1)

        # 获取形变，这里需要根据2D形变网络的具体实现来调整
        # 需要传入所有必要的参数：xyz, scaling, rotation, opacity, shs, time
        shs = torch.cat([self._features_dc, self._features_rest], dim=1)
        deform = self._deformation(self._xyz, self._scaling, self._rotation, self._opacity, shs, time)
        if isinstance(deform, tuple):
            xyz_deform = deform[0]  # 第一个返回值是位置形变
        else:
            xyz_deform = deform

        xyz = self._xyz + xyz_deform
        return xyz

    def compute_deformation(self, time):
        # 计算形变并累积到deformation_accum中
        xyz_deform = self.get_deformed_xyz(time) - self._xyz
        # 累积形变量（使用L2范数）
        self._deformation_accum += torch.abs(xyz_deform)
        return xyz_deform
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity) 
        if self.use_mip_filter:
            scales = self.scaling_activation(self._scaling)
            
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            
            scales_after_square = scales_square + torch.square(self.mip_filter) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            opacity = opacity * coef[..., None]
        return opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        print(scales)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 初始化deformation相关变量
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        
    def create_from_parameters(self, _means, _scales, _quaternions, _colors, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = _means
        fused_color = RGB2SH(_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.log(_scales)
        rots = _quaternions

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.00001)

        print("mean dist", dist2.mean())
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 初始化deformation相关变量
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('mip_filter')
        return l
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # ---------- 1. 把球谐 0 阶转成 RGB ----------
        # f_dc: (N, 1, 3)  -> 直接当颜色用
        rgb = self._features_dc.detach()[:, 0, :].cpu().numpy()   # (N, 3)
        rgb = np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)               # SH 系数 [-1,1]->[0,1]
        print(rgb)
        # ---------- 2. 其余属性保持原样 ----------
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.use_mip_filter:
            mip_filter = self.mip_filter.detach().cpu().numpy()
        # 3. 构造属性列表（把 rgb 插到 xyz 后面即可）
        dtype_full = [
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')] + \
                    [(name, 'f4') for name in self.construct_list_of_attributes()]
        if self.use_mip_filter:
            attributes = np.concatenate(((rgb * 255).astype(np.uint8), xyz, normals, f_dc, f_rest, opacities, scale, rotation, mip_filter), axis=1)
        else:
            attributes = np.concatenate(((rgb * 255).astype(np.uint8), xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 4. 拼接到一起
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    def save_ply_(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        if self.use_mip_filter:
            mip_filter = self.mip_filter.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_mip_filter:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, mip_filter), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "deformation.pth"), map_location="cuda")
        
        # --- 新增：处理尺寸不匹配的 aabb ---
        key_to_fix = "deformation_net.grid.aabb"
        if key_to_fix in weight_dict:
            checkpoint_shape = weight_dict[key_to_fix].shape
            model_shape = self._deformation.state_dict()[key_to_fix].shape
            aabb = weight_dict[key_to_fix]
            if checkpoint_shape != model_shape:
                print(f"Warning: Size mismatch for {key_to_fix}. Expected {model_shape}, got {checkpoint_shape}. Skipping this key.")
                del weight_dict[key_to_fix] # 删掉这个 key，不加载它
        # ----------------------------------

        # 使用 strict=False 也可以防止其他潜在的小差异导致崩溃
        self._deformation.load_state_dict(weight_dict,strict=False) 
        self._deformation = self._deformation.to("cuda")
        #self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation.deformation_net.set_aabb(aabb[0],aabb[1])
    def save_deformation(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
        
    @torch.no_grad()
    def get_tetra_points(
        self, 
        downsample_ratio : float = None, 
        gaussian_flatness : float = 1e-3, 
        return_idx : bool = False,
        points_idx : torch.Tensor = None,
        xyz : torch.Tensor = None,
        scaling : torch.Tensor = None,
        rotation : torch.Tensor = None,
    ):
        import trimesh
        M = trimesh.creation.box()
        M.vertices *= 2
        
        xyz = self.get_xyz if xyz is None else xyz
        scaling = self.get_scaling if scaling is None else scaling
        rotation = self._rotation if rotation is None else rotation
        rots = build_rotation(rotation)
        scales_3d = torch.nn.functional.pad(
            scaling,
            (0, 1), 
            mode="constant", 
            value=gaussian_flatness,
        )
        print(f"[INFO] Padding 2D scaling with {gaussian_flatness} for tetra points: {scales_3d[0]}")
        
        if (downsample_ratio is None) and (points_idx is None):
            scale = scales_3d * 3. # TODO test
            # filter points with small opacity for bicycle scene
            # opacity = self.get_opacity_with_3D_filter
            # mask = (opacity > 0.1).squeeze(-1)
            # xyz = xyz[mask]
            # scale = scale[mask]
            # rots = rots[mask]
        else:
            if points_idx is None:
                print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                xyz_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:int(xyz.shape[0] * downsample_ratio)]
                xyz = xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
            else:
                downsample_ratio = len(points_idx) / len(self.get_xyz)
                xyz_idx = points_idx
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
                
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        if return_idx:
            if downsample_ratio is None:
                print("[WARNING] return_idx might not be needed when downsample_ratio is None")
                xyz_idx = torch.arange(self.get_xyz.shape[0])
            return vertices, vertices_scale, xyz_idx
        else:
            return vertices, vertices_scale
    
    def set_mip_filter(self, use_mip_filter: bool):
        self.use_mip_filter = use_mip_filter
    
    @torch.no_grad()
    def compute_mip_filter(self, cameras, znear=0.2, filter_variance=0.2):
        # Set the flag to use the mip filter
        if not self.use_mip_filter:
            print("[WARNING] Computing mip filter but mip filter is currently disabled.")
        
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # We should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > znear
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        mip_filter = distance / focal_length * (filter_variance ** 0.5)
        self.mip_filter = mip_filter[..., None]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].copy()
        
        if "mip_filter" in [p.name for p in plydata.elements[0].properties]:
            mip_filter = np.asarray(plydata.elements[0]["mip_filter"])[..., np.newaxis].copy()
            use_mip_filter = True
            self.set_mip_filter(use_mip_filter)
            print("[INFO] Loading mip filter from ply file.")
        else:
            print("[INFO] No mip filter found in ply file.")
            use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        #opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if use_mip_filter:
            self.mip_filter = torch.tensor(mip_filter, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            mask = mask.to(group['params'][0].device)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask

        # 计算被prune的点的比例
        num_pruned = mask.sum().item()
        num_total = mask.shape[0]
        prune_ratio = num_pruned / num_total

        # 如果prune掉了超过一半的点，保存被删除的点为点云
        if prune_ratio > 0.5:
            print(f"[WARNING] Pruning {prune_ratio*100:.2f}% of points ({num_pruned}/{num_total})")
            print(f"[INFO] Saving pruned points to point cloud...")

            # 获取被prune掉的点的数据
            pruned_xyz = self._xyz[mask].detach().cpu().numpy()
            pruned_colors = self._features_dc[mask, 0, :].detach().cpu().numpy()
            pruned_colors = np.clip((pruned_colors + 1.0) * 0.5, 0.0, 1.0)  # SH系数转RGB
            pruned_opacity = self.opacity_activation(self._opacity[mask]).detach().cpu().numpy()

            # 保存为ply文件
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pruned_xyz)
            pcd.colors = o3d.utility.Vector3dVector(pruned_colors)

            # 生成文件名
            
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pruned_points_{timestamp}_ratio_{prune_ratio*100:.1f}.ply"

            save_path = os.path.join("pruned_points", filename)
            mkdir_p(os.path.dirname(save_path))
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"[INFO] Pruned points saved to: {save_path}")

            # 也保存不透明度信息到npy文件
            np.save(save_path.replace('.ply', '_opacity.npy'), pruned_opacity)

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #print(group,len(group["params"]))
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 处理deformation相关变量
        # if new_deformation_table is None:
        #     new_deformation_table = torch.gt(torch.ones((new_xyz.shape[0]),device="cuda"),0)
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, pts_num, stage = "coarse"):
        print(extent)
        use_mip_filter = self.use_mip_filter
        if use_mip_filter:
            self.set_mip_filter(False)
            
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        pts_num = self.get_xyz.shape[0]
        if pts_num < 360000:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
        pts_num = self.get_xyz.shape[0]
        if pts_num > 300000:
            if stage == "fine":
                adjusted_opacity = self.get_opacity
                motion_score = self.compute_average_displacement()
                motion_score = motion_score.unsqueeze(-1)
                print("use fine motion score")
                low_opacity_mask = (self.get_opacity < min_opacity).squeeze()
                adjusted_opacity = self.get_opacity.clone()
                #print(adjusted_opacity[low_opacity_mask].shape, motion_score[low_opacity_mask].shape)
                adjusted_opacity[low_opacity_mask] = adjusted_opacity[low_opacity_mask] * (motion_score[low_opacity_mask] + 1.0)
            
            else:
                motion_score = torch.ones_like(self.get_opacity)
                print("use coarse motion score")
                # 确保维度匹配，避免广播导致内存爆炸
                # get_opacity: (N, 1), motion_score: (N,)
                # 将 motion_score 扩展为 (N, 1) 以匹配
                # (N,) -> (N, 1)
                adjusted_opacity = self.get_opacity
                # 先判断哪些点的opacity小于阈值
                
                # 对于低opacity的点，乘以motion_score后再判断一次
            
            #adjusted_opacity = adjusted_opacity * (motion_score + 0.1)
            
            prune_mask = (adjusted_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                #prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                prune_mask = torch.logical_or(prune_mask, big_points_vs)

                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        if use_mip_filter:
            self.set_mip_filter(True)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # 根据形变累积量更新deformation table
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)

    def print_deformation_weight_grad(self):
        # 打印形变网络的梯度信息
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def compute_average_displacement(self, n_samples=10):
        """
        计算所有高斯点在n个时间戳上相对于canonical scene的平均位移

        Args:
            n_samples: 在[0, 1]之间采样的时间点数量

        Returns:
            avg_displacements: (N,) tensor，每个高斯点的平均位移（L2范数）
        """
        # 在[0, 1]之间均匀采样n个时间点
        timestamps = torch.linspace(0, 1, n_samples, device=self._xyz.device)

        # 获取canonical位置（t=0时的位置，或者直接用_xyz）
        canonical_xyz = self._xyz  # (N, 3)
        N = canonical_xyz.shape[0]
        print(f"N: {N}")
        # 累积所有时间点的位移
        total_displacement = torch.zeros(N, device=self._xyz.device)

        with torch.no_grad():
            for t in timestamps:
                # 将标量时间转换为tensor，并扩展到所有高斯点
                t_expanded = t.unsqueeze(0).expand(N, 1)  # (N, 1)

                # 获取当前时间的变形后位置
                deformed_xyz = self.get_deformed_xyz(t_expanded)  # (N, 3)

                # 计算位移（L2范数）
                displacement = torch.norm(deformed_xyz - canonical_xyz, dim=1)  # (N,)

                # 累积
                total_displacement += displacement

        # 计算平均位移
        avg_displacements = total_displacement / n_samples

        # 打印统计信息
        max_displacement = avg_displacements.max().item()
        mean_displacement = avg_displacements.mean().item()
        min_displacement = avg_displacements.min().item()
        print(f"[Displacement Statistics]")
        print(f"  Number of samples: {n_samples}")
        print(f"  Number of Gaussians: {N}")
        print(f"  Maximum average displacement: {max_displacement:.6f}")
        print(f"  Mean average displacement: {mean_displacement:.6f}")
        print(f"  Minimum average displacement: {min_displacement:.6f}")
        return avg_displacements
