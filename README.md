# PhysPRP

PhysPRP 复用 REST3D、PIXIE 和 PAC-NeRF，目标链路为：单帧 RGB 重建 → 非均匀物性场 → 可微 MPM → NeRF 视频渲染与训练。

当前已按功能阶段整理可复用源码，详见 [`model/README.md`](model/README.md)。

静态输入可通过 `PACNeRFStaticRGB(scene_dir, view_id, frame_id)` 从 PAC-NeRF 的 `all_data.json` 中选择指定视角和帧，输出 `[3, H, W]` 的归一化 RGB、相机外参与内参。

主流程入口为 `python main.py pipeline.example.json`。REST3D 输出 canonical OBJ，PIXIE 复用自身 Blender 入口合成多视角并输出物性 NPZ，PAC-NeRF 将该文件映射到已有的逐粒子物性参数后训练和渲染。
