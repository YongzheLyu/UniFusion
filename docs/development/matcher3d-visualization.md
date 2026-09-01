# Matcher3D 可视化指南

这个文档介绍了如何使用 `Matcher3D` 类的可视化功能来分析不同图表（深度图）之间的几何匹配关系。

## 背景知识

`Matcher3D` 类用于检测多个深度图之间的几何一致性：

- **输入**：多个深度图（每个图表对应一个相机视角）
- **输出**：`reference_matches[i, j, h, w]` 表示图表i中(h,w)位置的点是否与图表j中的对应点几何一致

## 可视化功能

### 1. 单个图表的匹配可视化

```python
matcher.visualize_matches_per_chart(chart_idx=0, save_path="matches_chart_0.png")
```

**功能**：显示指定图表与其他所有图表的匹配情况

- **输入参数**：
  - `chart_idx`: 要可视化的图表索引
  - `save_path`: 保存路径（可选）
  - `figsize`: 图像尺寸（默认(15, 10)）

- **输出**：一个包含多个子图的图像，每个子图显示该图表与另一个图表的匹配结果
  - 绿色：匹配成功
  - 红色：匹配失败
  - 标题显示匹配的像素数量和百分比

### 2. 共识匹配可视化

```python
matcher.visualize_consensus_matches(min_consensus=2, save_path="consensus.png")
```

**功能**：显示多个图表一致匹配的像素点

- **输入参数**：
  - `min_consensus`: 最少需要多少个图表一致匹配（默认2）
  - `save_path`: 保存路径（可选）

- **输出**：两个子图
  - **上方**：共识匹配的二值图（蓝色表示共识匹配）
  - **下方**：每个像素被多少个图表匹配的热力图

### 3. 匹配统计信息

```python
matcher.visualize_matching_statistics(save_path="statistics.png")
```

**功能**：显示匹配的整体统计信息

- **输出**：三个子图
  1. **匹配率矩阵**：显示每对图表之间的匹配率
  2. **平均匹配数**：每个图表的平均匹配数量
  3. **匹配分布**：匹配数量的直方图

### 4. 一键生成所有可视化

```python
matcher.visualize_all(output_dir="./visualizations", min_consensus=2)
```

**功能**：生成上述所有类型的可视化

## 使用示例

### 在 parallel_aligner_temporal.py 中使用

```python
# 在Matcher3D创建和匹配后添加
if verbose:
    vis_dir = os.path.join(self.model_path, f'matcher_visualizations_timestamp_{timestamp_idx}')
    try:
        matcher_t.visualize_all(output_dir=vis_dir, min_consensus=2)
        print(f"[INFO] Saved matcher visualizations for timestamp {timestamp_idx} to: {vis_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to generate visualizations for timestamp {timestamp_idx}: {e}")
```

### 独立使用

```python
from matcha.dm_modules.matcher_3d import Matcher3D

# 1. 创建matcher并运行匹配
matcher = Matcher3D(cameras=cameras, reference_depths=depths)
matcher.match(matching_threshold)

# 2. 生成可视化
matcher.visualize_all(output_dir="./my_visualizations")

# 或者单独使用
matcher.visualize_matches_per_chart(chart_idx=0)
matcher.visualize_consensus_matches(min_consensus=3)
matcher.visualize_matching_statistics()
```

## 输出文件说明

运行 `visualize_all()` 后会生成：

```
output_dir/
├── matches_chart_0.png      # 图表0与其他图表的匹配
├── matches_chart_1.png      # 图表1与其他图表的匹配
├── ...
├── consensus_matches.png    # 共识匹配分析
└── matching_statistics.png  # 统计信息
```

## 运行演示

使用提供的示例脚本：

```bash
# 创建示例matcher并保存
python matcher_visualization_example.py --create_example

# 生成可视化（使用示例数据）
python matcher_visualization_example.py --output_dir ./demo_visualizations

# 或使用真实保存的matcher
python matcher_visualization_example.py --matcher_path /path/to/matcher.pkl --output_dir ./real_visualizations
```

## 解读结果

### 匹配率矩阵
- **对角线**：自匹配（通常为1.0，因为点与自己总是匹配）
- **非对角线**：不同图表之间的匹配率
- **高匹配率**：表示这些图表在几何上高度一致
- **低匹配率**：可能表示遮挡、运动或深度噪声

### 共识匹配
- **高共识区域**：多个图表一致观察到的几何结构（可靠的几何信息）
- **低共识区域**：可能存在遮挡、噪声或不准确的深度测量

### 匹配统计
- **平均匹配数**：每个图表的整体一致性指标
- **匹配分布**：了解匹配关系的多样性

## 注意事项

1. **内存使用**：可视化大尺寸图像时注意内存使用
2. **文件大小**：高分辨率可视化会生成大文件
3. **颜色编码**：
   - 绿色/蓝色：匹配成功
   - 红色：匹配失败
   - 热力图：从蓝到黄到红表示从低到高

## 故障排除

- **ImportError**: 确保matplotlib已安装
- **MemoryError**: 减小图像尺寸或降低分辨率
- **Empty plots**: 检查matcher是否已调用`match()`方法
