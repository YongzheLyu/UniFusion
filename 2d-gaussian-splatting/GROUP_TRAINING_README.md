# 分组训练功能说明

## 功能概述

该功能允许将相机帧按照指定间隔分为多个组，每组使用独立的Gaussian Splatting模型进行训练。训练过程是顺序进行的，后续组会从前一组的最后一帧静态点云初始化，从而实现增量式训练。

## 主要特性

1. **分组训练**: 将所有训练相机按帧间隔分组
2. **点云继承**: 每组从前一组的静态点云初始化（除了第一组）
3. **独立模型**: 每组训练独立的Gaussian模型并保存到独立目录
4. **内存优化**: 训练完每组后自动清理内存

## 使用方法

### 基本命令

```bash
python train_with_charts_.py \
  --source_path /path/to/your/data \
  --model_path /path/to/output \
  --enable_group_training \
  --frame_interval 30 \
  --frames 16 \
  --iterations 7000 \
  --coarse_iterations 3000
```

### 参数说明

- `--enable_group_training`: 启用分组训练模式（必须）
- `--frame_interval`: 每组的帧数间隔，默认30帧
  - 例如：设置为30时，会将相机分为[0-29], [30-59], [60-89]...等组
- `--frames`: 每个相机的时间帧数，默认16
- `--iterations`: fine阶段的训练迭代次数
- `--coarse_iterations`: coarse阶段的训练迭代次数（可选）

### 示例场景

#### 场景1: 训练0-30帧，然后30-60帧

```bash
python train_with_charts_.py \
  --source_path ./data/my_scene \
  --model_path ./output/my_scene_grouped \
  --enable_group_training \
  --frame_interval 30 \
  --iterations 7000 \
  --coarse_iterations 3000
```

这将：
1. 首先训练第0-29帧（Group 1），使用charts数据初始化
2. 然后训练第30-59帧（Group 2），使用Group 1的最后状态初始化
3. 以此类推...

#### 场景2: 更小的组，每组20帧

```bash
python train_with_charts_.py \
  --source_path ./data/my_scene \
  --model_path ./output/my_scene_small_groups \
  --enable_group_training \
  --frame_interval 20 \
  --iterations 5000
```

## 输出结构

训练完成后，输出目录结构如下：

```
output/my_scene_grouped/
├── group_1/
│   ├── point_cloud/
│   │   └── iteration_7000/
│   ├── cameras.json
│   └── cfg_args
├── group_2/
│   ├── point_cloud/
│   │   └── iteration_7000/
│   ├── cameras.json
│   └── cfg_args
├── group_3/
│   └── ...
└── ...
```

每个`group_X`目录包含该组的完整训练结果。

## 工作原理

### 1. 相机分组

程序会自动获取所有训练相机，并按`frame_interval`参数分组：
- Group 1: cameras[0:30]
- Group 2: cameras[30:60]
- Group 3: cameras[60:90]
- ...

### 2. 初始化策略

- **第一组 (Group 1)**:
  - 使用原始的charts数据初始化
  - 通过`get_gaussian_parameters_from_charts_data`函数生成初始高斯参数

- **后续组 (Group 2+)**:
  - 从前一组的训练结果继承点云
  - 复制以下参数：
    - 位置 (xyz)
    - 缩放 (scaling)
    - 旋转 (rotation/quaternions)
    - 颜色 (colors from SH features)

### 3. 训练流程

对每个组：
1. 创建新的GaussianModel实例
2. 初始化（使用charts或继承点云）
3. 训练coarse阶段（如果指定）
4. 训练fine阶段
5. 保存模型
6. 清理内存并继续下一组

## 注意事项

### 内存管理

- 每组训练完成后会自动调用`torch.cuda.empty_cache()`和`gc.collect()`
- 建议根据GPU内存大小调整`frame_interval`
- 较小的组可以降低单次训练的内存需求

### 训练时间

- 总训练时间 ≈ 单组训练时间 × 组数
- 可以通过减少`--iterations`来加速训练

### 数据要求

- 需要预处理的charts数据: `charts_data.npz`
- 需要预处理的priors（如果使用`--preprocessed_priors_dir`）
- 相机数据必须按时间顺序排列

## 故障排除

### 问题1: 内存不足

**解决方案**: 减小`frame_interval`，例如从30减到20或15

```bash
--frame_interval 15
```

### 问题2: 点云质量下降

**解决方案**:
- 增加每组的训练迭代次数
- 调整coarse和fine阶段的迭代比例

```bash
--iterations 10000 --coarse_iterations 5000
```

### 问题3: 组间过渡不平滑

**解决方案**:
- 减小frame_interval，增加组间重叠
- 或在后续组训练时增加正则化权重

## 与原始训练的对比

| 特性 | 原始训练 | 分组训练 |
|------|---------|---------|
| 内存需求 | 高 | 可控 |
| 训练时间 | 较短 | 较长（顺序训练） |
| 灵活性 | 低 | 高（可针对不同时间段调整） |
| 模型数量 | 1个 | 多个（每组1个） |
| 适用场景 | 短序列 | 长序列、大场景 |

## 高级用法

### 禁用分组训练

如果不使用分组训练，只需不添加`--enable_group_training`标志即可恢复原始训练模式：

```bash
python train_with_charts_.py \
  --source_path ./data/my_scene \
  --model_path ./output/my_scene_normal \
  --iterations 7000
```

### 自定义每组的训练参数

目前所有组使用相同的训练参数。如需为不同组使用不同参数，可以：

1. 分别运行多次训练
2. 使用第一组的输出作为第二组的checkpoint
3. 手动调整每组的参数

## 技术细节

### 点云继承实现

```python
# 从前一组复制参数
_means = prev_group_gaussians.get_xyz.detach().clone()
_scales = prev_group_gaussians.get_scaling.detach().clone()
_quaternions = prev_group_gaussians.get_rotation.detach().clone()
_colors = SH2RGB(prev_group_gaussians._features_dc.detach().clone()[:, 0])

# 创建新模型
group_gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, spatial_lr_scale)
```

### 相机过滤

```python
# 筛选当前组的相机
scene.train_cameras = all_train_cameras[start_idx:end_idx]
```

## 版本信息

- 添加日期: 2025-12-03
- 基于代码: train_with_charts_.py
- 兼容性: 需要完整的MAtCha/2d-gaussian-splatting环境

## 联系与支持

如有问题或建议，请查看项目文档或提交issue。
