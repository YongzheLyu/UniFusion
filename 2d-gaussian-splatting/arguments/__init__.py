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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']

        # Migrated from all_people.py
        self.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 16,
            'resolution': [64, 64, 64, 100]
        }
        self.multires = [1, 2, 4]
        self.defor_depth = 1
        self.net_width = 128
        self.plane_tv_weight = 0.0002
        self.time_smoothness_weight = 0.001
        self.l1_time_planes = 0.0001
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_do = True
        self.no_dshs = False
        self.no_ds = False
        self.no_dr=False
        self.empty_voxel = False
        self.render_process = False
        self.static_mlp = False
        self.bounds = 10
        self.timebase_pe = 4 # useless
        self.grid_pe=0 # useless
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.apply_rotation=False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100
        
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        #尝试改变max steps 11.05
        self.position_lr_max_steps = 20_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0  # 0.0 previously. Changed to 100.0 to match the paper.
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        # self.plane_tv_weight = 0.0002
        # self.time_smoothness_weight = 0.001
        # self.l1_time_planes = 0.0001
        
        
        self.densification_interval = 100
        #self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        #self.iterations = 30_000
        self.coarse_iterations = 3000
        # self.position_lr_init = 0.00016
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        # self.feature_lr = 0.0025
        # self.opacity_lr = 0.05
        # self.scaling_lr = 0.005
        # self.rotation_lr = 0.001
        # self.percent_dense = 0.01
        # self.lambda_dssim = 0
        # self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 60000
       
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size=1
        self.add_point=False
        self.freeze_gaussians_in_fine = True  # If True, freeze Gaussian parameters (xyz, opacity, scaling, rotation, features) for first 1000 iterations in fine stage
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found at")
        pass

    # 安全地解析配置文件，处理包含Python对象的情况
    try:
        args_cfgfile = eval(cfgfile_string)
    except (SyntaxError, NameError) as e:
        print(f"Warning: Failed to parse config file with eval(): {e}")
        print("Attempting to clean the config string...")

        # 清理配置文件字符串，移除无法解析的Python对象
        import re
        # 移除类似 _gaussians=<scene.gaussian_model.GaussianModel object at 0x...> 的模式
        cleaned_string = re.sub(r',\s*[_a-zA-Z]\w*=<[^>]+object\s+at\s+0x[0-9a-fA-F]+>', '', cfgfile_string)
        # 移除类似 <[^>]+object at 0x[0-9a-fA-F]+> 的模式（开头的情况）
        cleaned_string = re.sub(r'[_a-zA-Z]\w*=<[^>]+object\s+at\s+0x[0-9a-fA-F]+>,?\s*', '', cleaned_string)

        try:
            args_cfgfile = eval(cleaned_string)
            print("Successfully parsed cleaned config file")
        except Exception as e2:
            print(f"Failed to parse cleaned config: {e2}")
            print("Using default empty Namespace")
            args_cfgfile = Namespace()

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        # Preserve config-file values when an optional CLI argument was omitted
        # (and therefore parsed as None), but still expose newly-added parser
        # arguments that are absent from older saved cfg_args files.
        if v is not None or k not in merged_dict:
            merged_dict[k] = v
    return Namespace(**merged_dict)
