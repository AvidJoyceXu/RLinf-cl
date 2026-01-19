# 问题诊断与解决方案

## 问题：演示数据文件找不到

### 错误信息

```
Demo path: /opt/venv/openvla-oft/libero/libero/libero/../datasets/libero_spatial/...
Warning: Demo file not found: ...
Error: Failed to collect state samples from task 0!
```

### 问题原因

1. **路径解析错误**：`get_libero_path("datasets")` 返回的路径可能包含相对路径（如 `../datasets`），导致路径解析失败
2. **LIBERO 数据集未安装**：演示数据文件可能未正确安装或路径配置不正确
3. **环境变量未设置**：`LIBERO_PATH` 环境变量可能未正确设置

### 解决方案

#### 方案 1：使用 `base_rollout` 方法（推荐）

**优点**：
- 不需要演示数据文件
- 更符合实际 RL 训练时的状态分布
- 分析结果更准确

**使用方法**：

修改 `run_correction_field_analysis.sh`：
```bash
STATE_METHOD="base_rollout"  # 改为 base_rollout
```

或直接使用命令行：
```bash
python analyze_correction_vector_field.py \
    --config configs/eval_lora_config.yaml \
    --task_i 0 \
    --task_j 1 \
    --state_method base_rollout \
    --max_demos 5
```

#### 方案 2：修复演示数据路径

**步骤 1：检查 LIBERO 数据集路径**

```bash
python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))"
```

**步骤 2：设置环境变量**

如果路径不正确，设置 `LIBERO_PATH` 环境变量：

```bash
export LIBERO_PATH=/path/to/libero
```

**步骤 3：确保数据集已安装**

检查演示数据文件是否存在：
```bash
ls $LIBERO_PATH/libero/datasets/libero_spatial/*.hdf5
```

**步骤 4：使用绝对路径**

如果相对路径有问题，可以在代码中手动指定数据集路径。

### 代码修复

我已经修复了路径解析问题：

1. **路径规范化**：使用 `os.path.normpath()` 和 `os.path.abspath()` 规范化路径
2. **更好的错误提示**：当文件找不到时，提供详细的诊断信息和解决方案建议
3. **路径调试信息**：打印路径解析的详细信息，便于调试

### 修复后的行为

当演示数据文件找不到时，脚本会：

1. 打印详细的错误信息
2. 显示路径解析的详细信息
3. 提供解决方案建议（推荐使用 `base_rollout`）
4. 显示示例命令

### 验证修复

运行脚本后，如果仍然找不到文件，会看到类似输出：

```
Demo path: /resolved/absolute/path/to/demo.hdf5
  (resolved from: /path/to/datasets + relative/path/to/demo.hdf5)
Warning: Demo file not found: /resolved/absolute/path/to/demo.hdf5
  Please check:
    1. LIBERO datasets are installed
    2. LIBERO_PATH environment variable is set correctly
    3. Dataset file exists at the expected location

  Alternative: Use --state_method base_rollout instead of demo/both_demo
```

### 推荐做法

**对于大多数情况，推荐使用 `base_rollout` 方法**：

1. **更准确**：使用 Base Model rollout 收集的状态更符合实际 RL 训练时的状态分布
2. **更简单**：不需要安装和配置演示数据
3. **更可靠**：不依赖外部数据文件

### 其他常见问题

#### Q: 为什么 `base_rollout` 方法更推荐？

A: 因为 RL 实际看到的状态是由 Base Model 的行为决定的。只有在 Base Model 实际会访问到的状态上，Residual RL 才有修正的机会。使用 `base_rollout` 可以更准确地反映实际训练时的状态分布。

#### Q: 如何检查 LIBERO 数据集是否正确安装？

A: 运行以下命令：
```bash
python -c "
from libero.libero import get_libero_path
import os
datasets_path = get_libero_path('datasets')
print(f'Datasets path: {datasets_path}')
print(f'Path exists: {os.path.exists(datasets_path)}')
if os.path.exists(datasets_path):
    files = os.listdir(datasets_path)
    print(f'Files in datasets: {files[:5]}...')
"
```

#### Q: 如果必须使用 `demo` 方法怎么办？

A: 确保：
1. LIBERO 数据集已正确安装
2. `LIBERO_PATH` 环境变量正确设置
3. 演示数据文件存在于预期位置
4. 路径解析正确（已通过代码修复）

### 总结

- **推荐使用 `base_rollout` 方法**，不需要演示数据文件
- 如果必须使用 `demo` 方法，确保 LIBERO 数据集正确安装和配置
- 代码已修复路径解析问题，提供更好的错误提示

