# Azul 训练说明（DinoBoard Intelligence）

本目录是 Azul 的训练入口与游戏专属逻辑；通用训练骨架已沉淀到 `general/train`。

当前完整闭环：

1. C++ 自博弈采样（并行 worker，支持 `thread/process` 池）
2. 增量样本进入 replay buffer（内存缓存）
3. Torch policy/value 训练并导出 `candidate_model.onnx`
4. gating 与 benchmark（`netmcts`）
5. best/latest/history_best 管理与晋升

## 配置来源（单一真源）

- 默认配置来自 `train_config.example.json`。
- `plugin.py` 的 `_default_config` 会直接读取该文件，不再维护内嵌默认字典。
- CLI 参数会覆盖配置文件中的对应字段（通过通用 CLI 层完成）。

## 哪些在 game，哪些在 general

- 放在 `games/azul/train`（游戏专属）：
  - C++ 后端绑定与动作策略：`plugin.py`（当前已合并；也可按需拆回 `cpp_training_backend.py`）
  - 自博弈/对战调用：`plugin.py`
  - Azul 标签策略（margin/phase）：`plugin.py` 中 hooks
  - Azul 维度与标签常量：`constants.py`

- 放在 `general/train`（跨游戏复用）：
  - pipeline 主流程与 step loop
  - 配置模型与 CLI 解析
  - 通用 Torch 训练骨架（如 `torch_simple_trainer.py`）
  - PVNet 构建/ONNX 导出、checkpoint、runtime cache、样本工具等

## 运行方式

确保 `cpp_azul_engine_v7` 已编译且当前环境可导入后执行：

```powershell
python -m games.azul.train.plugin `
  --config "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/azul/train/train_config.example.json" `
  --output "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/azul/train/out/job_001"
```

或仅指定输出目录，走默认配置文件：

```powershell
python -m games.azul.train.plugin `
  --output "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/azul/train/out/job_default"
```

## 主要产物

- `training_config.json`
- `job_status.json`
- `artifacts/selfplay_summary.json`
- `artifacts/models/candidate_model.onnx`
- `artifacts/models/best_model.onnx`
- `artifacts/models/latest_model.onnx`
- `artifacts/gating_summary.json`
- `artifacts/candidate_manifest.json`
