# DinoBoard Intelligence - General 架构说明

`general/` 是跨游戏复用层，只放通用编排、通用协议、通用工具，不放游戏规则语义。

## 总体分层

- `include/` + `src/`
  - C++ 通用能力：核心类型、规则接口、NetMCTS、ONNX 评估器。
  - NetMCTS 根节点温度采样统一由 `search/select_action_from_visits(...)` 提供，
    游戏侧应复用该函数，禁止重复实现。
  - 单局内温度线性衰减可复用 `search/temperature_schedule.h`（C++ 通用工具）。
  - 根节点 Dirichlet 噪声窗口与参数归一可复用 `search/root_noise.h`（C++ 通用工具）。
  - `search_options` 公共字段解析可复用 `search/search_options_common.h`（C++ 通用工具）。
- `train/`
  - 训练编排主干（selfplay -> train -> periodic eval -> gating）与通用训练工具。
- `debug/`
  - 单游戏调试服务通用壳（FastAPI app factory、运行时服务、C++ 扩展构建辅助）。
- `debug_hub/`
  - 多游戏调试聚合入口。
- `web/`
  - 调试前端通用组件。
- `scripts/`
  - 通用脚本（环境准备、训练启动、调试启动）。

## Debug Service 通用约定

- 调试环境与启动统一走脚本：
  - `general/scripts/setup_game_env.ps1`
  - `general/scripts/run_game_debug.ps1`
- 每个游戏目录下的 `debug_service/setup_env.ps1` 与 `debug_service/run_debug.ps1`
  仅保留参数化薄包装，避免重复维护环境逻辑。
- 调试服务共享虚拟环境：`DinoBoard Intelligence/.venv_debug_shared`。
- 支持统一 requirements hash 复用、可选 ONNX Runtime 安装参数、统一 uvicorn 启动方式。
- 会话生命周期统一为“默认单活会话”（新开局前清理旧会话），并提供显式删除接口
  `DELETE /api/v1/games/{session_id}`。
- 通用清理脚本：`general/scripts/clean_artifacts.ps1`。

## 训练架构（当前）

### 编排层

- `train/pipeline.py`
  - 顶层入口 `run_train_job(...)`，负责阶段组织与最终汇总。
- `train/pipeline_selfplay_loop.py`
  - `SelfplayLoopContext` 组装后的编排器，串联 warm start 与 step loop。
- `train/pipeline_warm_start.py`
  - warm start 自博弈与 warm trainer pass。
- `train/pipeline_step_loop.py`
  - 主循环每 step：selfplay、trainer、periodic eval。
- `train/pipeline_periodic_eval.py`
  - 周期评估（candidate 对 benchmark/history_best）。
- `train/pipeline_finalize.py`
  - 自博弈汇总产物、最终 gating、candidate 晋升。

### 复用辅助

- `train/pipeline_support.py`
  - 通用 seed、线程/进程池、arena 对战、policy 组装辅助（selfplay/eval）。
  - 内置可选“单局内温度线性衰减插件”注入：当
    `selfplay.exploration.temperature_decay_plies > 0` 时，自动向
    `search_options` 注入 `temperature_initial/temperature_final/temperature_decay_plies`。
- `train/config.py`
  - `TrainJobConfig` 及子配置结构。
- `train/mcts_schedule.py`
  - simulations 调度规则（线性区间固定为 step 0 到总 steps）。
- `train/policy_bridge.py`
  - `PolicyConfig` 到 `search_options` 的默认桥接。
- `train/extensions.py`
  - `TrainPipelineHooks` 扩展点（policy hook、sample label hook 等）。

### 训练实现层

- `train/torch_simple_trainer.py`
  - 通用简单离散动作 Trainer（增量样本 -> replay cache -> ONNX/PT 导出）。
- `train/torch_pvnet.py`
  - 通用 PVNet 搭建与 ONNX 导出。
- `train/torch_runtime.py` / `train/torch_checkpoint.py`
  - Torch 运行时缓存与 checkpoint。
- `train/torch_sample_extractors.py` / `train/policy_target_utils.py`
  - 样本抽取与 policy target 归一化工具。
- `train/trainer_interface.py`
  - 训练器契约：`run(config, artifacts_dir, *, resume_checkpoint_path, step_index, total_steps, incremental_samples) -> dict`。
  - 返回值采用 `dict`（与 Azul/Splendor 实现一致），由 pipeline 直接合并到 `train_summary`。

## 当前训练数据流（重要）

- 自博弈样本不再落盘 `selfplay_samples.jsonl`。
- 训练走纯增量内存路径：`selfplay -> incremental_samples -> trainer replay cache`。
- 默认保留的核心产物：`selfplay_summary.json`、`train_summary.json`、`gating_summary.json`、`models/{candidate,best,latest}`。
- `TrainJobConfig` 作为输入快照使用；运行态模型路径仅存于 pipeline `state`，不回写输入 `config`。

## 新游戏接入

为避免文档重复，完整落地步骤、最小文件集、验收标准统一维护在：

- `general/NEW_GAME_IMPLEMENTATION_CHECKLIST.md`

这里仅保留总原则：

- 新游戏入口统一为 `games/<game>/train/plugin.py` 与 `games/<game>/debug_service/plugin.py`。
- 通用样板优先复用 `general.train` 与 `general.debug` 的 loader / factory。
- 游戏语义实现放在 `games/<game>/...`，不要回灌到 `general/`。

## 设计边界

- `general/` 不承载任何具体游戏规则细节。
- 游戏语义只在 `games/<game>/...` 下实现，并通过 hooks/callback 注入。
- 若一个逻辑在两个以上游戏复用，优先上收到 `general/`。

