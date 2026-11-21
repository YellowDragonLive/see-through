# PSD Studio 变更说明

本地 Web UI + CLI 的 PSD 分层修整工具，服务于 See-through / Anime2.5DRig 的产物后处理。
当前只保留最新快照 **v2.4 turbo**，历史版本不再随仓库分发。

- 最新版本：v2.4 turbo（2026-11-21 13:12）
- 依赖：`numpy`、`opencv-python`、`Pillow`（无额外 pip 依赖）
- 入口：Windows 双击 `start_studio_win.bat`；命令行 `python psd_studio.py --help`

## 版本总览

| 版本 | 快照时间 | psd_studio.py | 主题 | 现状 |
| --- | --- | --- | --- | --- |
| v2.1 fast | 2026-11-21 12:50 | 49,611 B | 首个可用版：PSD 读写、CV 管线、排序、Web UI + CLI | 已移除 |
| v2.2 | 2026-11-21 12:59 | 53,969 B | 帽子/头饰按伪深度三明治拆分 | 已移除 |
| v2.3 | 2026-11-21 13:05 | 54,603 B | 修羽化泄漏、吞暗边、清碎弧 | 已移除 |
| v2.4 turbo | 2026-11-21 13:12 | 55,767 B | ROI 缓存、降采样统计、耗时回显、HTTP/1.1、预览限幅 | **保留** |

HTML 界面在 v2.2 起新增「帽子/头饰伪深度三明治拆分」开关，v2.3、v2.4 未再改动界面。

## v2.1 fast — 基线

- PSD 解析与写回（PackBits 解压/压缩、图层裁剪、合成）
- CV 管线：`clean_alpha_and_specks`、`decontaminate_edges`、`split_layer_by_depth`、`refine_neck_topwear`
- `sort_layers_by_mode` 多模式排序 + 解剖层级锁定
- `StudioSession` / `StudioHandler` 本地 Web UI，支持撤销、可见性、头部微移、导出 PSD

## v2.2 — 头饰三明治拆分

- 新增 `refine_headwear_sandwich()`：把 `headwear` 按逐像素伪深度对 `face` 拆成
  - `headwear_1`：深度大于面部中位数 → 帽后腔/内里，压在脸后面
  - `headwear_2`：深度不大于面部中位数 → 帽檐/前片，压在脸和刘海前面
- 触发门槛：与面部重叠 ≥ 300px、后腔占重叠 ≥ 12%、后腔连通域 ≥ 200px（清洗后 ≥ 300）
- 暗色抗锯齿内腔边沿吞并（7×7、亮度 < 95），feather 羽化，并向内 dilate 形成 underlap，避免细线穿脸
- 排序权重：`headwear_1` = 9.5（耳朵/脸之后、刘海之前），`headwear_2` = 24.0（最前）
- 层级锁定：`headwear_1` 置于 ears/face 之前，`headwear_2` 置于 face/front hair 之后
- `head` 区域框纳入 `headwear_1` / `headwear_2`；新增参数 `headwear_sandwich`（默认开）与界面开关

## v2.3 — 修三处遮罩缺陷

- **羽化泄漏**：模糊前把图层 alpha 为 0 的区域填 1.0（`bm_src`），防止高斯羽化把背景/脸的反向权重渗进前层；topwear 与 headwear 两个分支同步修
- **半透明暗边**：内腔外沿吞并半径 7×7 → 13×13，判据增加 `alpha < 96`，解决帽腔内沿残留的半透明细线
- **孤立碎弧**：`headwear_2` 中覆盖面部、面积 < 600px 的非主连通域直接清零

## v2.4 turbo — 性能

- `StudioSession` 新增 `_cached_rois` 与 `_refresh_rois_cache()`；`get_state(recompute_rois=False)` 复用缓存，加载/重置时失效
- `get_state` 统计降采样：`step = max(1, max(H, W) // 256)`，像素数按 `step²` 还原，bbox 按 step 对齐还原 —— 大图状态刷新明显变快
- `apply_pipeline` / `apply_sort_only` 用 `time.perf_counter()` 计时，并在日志尾部回显 `(N ms)`
- HTTP 服务设 `protocol_version = "HTTP/1.1"`（keep-alive），`address_string()` 直接返回客户端 IP
- 预览合成限幅：整图 768px、裁剪区域 1024px，降低前端解码与传输开销

## 版本管理约定

- 仓库只保留最新一份快照目录，旧快照不提交（v2.2、v2.3 为中间产物，从未入库）
- 发布新版本时：新建快照目录 → 更新本文件的版本总览与对应章节 → 删除旧快照
