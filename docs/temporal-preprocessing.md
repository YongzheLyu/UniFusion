# Temporal Charts Alignment - Data Preprocessing

这个文档说明了如何使用预处理脚本来加速时间序列图表对齐过程。

## 概述

原始的 `align_charts_temporal.py` 脚本在每次运行时都需要：
1. 为每个时间戳生成pointmap（包括运行DepthAnything模型）
2. 计算缩放因子和参考数据
3. 执行图表对齐训练

这个预处理方法将数据准备步骤分离出来，只需运行一次，然后可以多次运行对齐训练。

## 数据准备流程

### 1. 预处理数据 (只运行一次)

```bash
python scripts/preprocess_temporal_data.py \
    -d /path/to/data_directory \
    -o /path/to/preprocessed_data.pkl \
    --start_frame 0 \
    --end_frame 100 \
    --depthanything_encoder vitl
```

**参数说明：**
- `-d, --data_dir`: 包含 `frame_XXXXX` 子目录的基础目录
- `-o, --output_path`: 预处理数据的输出路径（默认: `{data_dir}/preprocessed_temporal_data.pkl`）
- `--start_frame`: 起始帧索引
- `--end_frame`: 结束帧索引
- `--depthanything_encoder`: DepthAnything编码器类型 (vitl, vitb, vits, vitg)

**预处理输出：**
- `temporal_scene_pms`: 每个时间戳的场景pointmap
- `temporal_sfm_datas`: SFM数据（包含xyz坐标、颜色、图像点映射）
- `temporal_mast3r_pms`: MASt3R pointmaps
- `temporal_reference_data`: 计算出的参考数据
- `temporal_mast3r_masks`: 可选的mask数据
- `scale_factor`: 缩放因子
- 配置文件和参数

### 2. 运行图表对齐 (可多次运行)

```bash
python scripts/align_charts_temporal_from_preprocessed.py \
    -p /path/to/preprocessed_data.pkl \
    -o /path/to/output_directory \
    --temporal_encoding_type learned \
    --temporal_encoding_dim 8
```

**参数说明：**
- `-p, --preprocessed_data`: 预处理数据文件的路径
- `-o, --output_path`: 对齐结果输出目录
- `--temporal_encoding_type`: 时间编码类型 (learned 或 positional)
- `--temporal_encoding_dim`: 时间特征维度

## 优势

1. **时间节省**: 预处理只需运行一次，避免重复的pointmap生成
2. **内存效率**: 可以分阶段管理GPU内存
3. **灵活性**: 可以尝试不同的对齐参数而不重新生成数据
4. **调试友好**: 数据准备和训练分离，便于调试

## 数据结构

预处理脚本保存的数据结构与原始 `align_charts_temporal.py` 完全兼容：

```python
preprocessed_data = {
    'temporal_scene_pms': [...],        # PointMapDepthAnything 对象列表
    'temporal_sfm_datas': [...],        # SFM数据字典列表
    'temporal_mast3r_pms': [...],       # PointMapMASt3R 对象列表
    'temporal_reference_data': [...],   # 参考数据张量列表
    'temporal_mast3r_masks': [...],     # Mask数据列表（可选）
    'temporal_frame_indices': [...],    # 帧索引列表
    'scale_factor': float,              # 缩放因子
    'pm_config': dict,                  # Pointmap配置
    'scene_config': dict,               # 场景配置
    'masking_config': dict,             # Mask配置
    'args': dict,                       # 预处理参数
}
```

## 使用建议

1. **大规模数据集**: 对于包含数百帧的数据集，预处理特别有用
2. **参数调优**: 使用预处理数据可以快速尝试不同的对齐参数
3. **内存管理**: 预处理脚本包含GPU内存清理逻辑，适合大场景处理

## 故障排除

- **内存不足**: 减少 `--max_frames` 或使用更小的 `max_img_size`
- **模型加载失败**: 检查 `depthanything_checkpoint_dir` 路径
- **帧目录不存在**: 确保数据目录结构正确（`frame_XXXXX/mast3r_sfm/`）
