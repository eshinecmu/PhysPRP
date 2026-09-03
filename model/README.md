# Model modules

目录按“处理阶段 / 来源项目”分两级，来源项目内部保持原目录结构，避免改写已有导入关系。

```text
model/
├── reconstruction/rest3d/  # 单帧 RGB → 3D 几何/高斯中间结果
├── material/pixie/         # 体素化 → 分割与连续物性预测
└── dynamics/pac_nerf/      # 粒子采样 → 可微 MPM → NeRF 渲染与训练
```

## 计划中的调用顺序

1. `reconstruction/rest3d/scripts/infer_scene3d.py` 调用 `rest3d/models/build_sam3dobj.py`，从 PAC-NeRF 数据集中选定视角的一帧 RGB 生成几何与高斯中间表示。
2. `material/pixie/pixie/voxel/` 构建体素特征；`third_party/Wavelet-Generation/trainer/inference_combined.py` 输出离散材料类别和连续、非均匀物性场。
3. `dynamics/pac_nerf/lib/nerf/dvgo.py` 完成密度场/粒子采样，`lib/engine/mpm_simulator.py` 与 `dynamic_observer.py` 执行可微 MPM。
4. `dynamics/pac_nerf/lib/pac_nerf.py` 将粒子状态映射回动态 NeRF；`train.py`、`test.py` 复用原训练和视频渲染链路。

## 来源版本

- REST3D: `9d683303d9163b50a09145929de2274023cf3449`
- PIXIE: `bfe198a6d6024f606f64acaf537237da761ebfec`
- PAC-NeRF: `b613048557d0648e885697ececbef80297defac0`

各模块保留其原始 `LICENSE`。目前仅归档可复用板块，尚未添加跨模型适配代码。
