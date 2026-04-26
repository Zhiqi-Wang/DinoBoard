# DinoBoard Intelligence - Azul 模块说明

`games/azul/` 承载 Azul 的游戏语义实现，并通过回调/接口接入 `general/` 通用框架。

## Azul 在整体架构里的位置

- `general/` 负责通用编排、通用协议、通用工具。
- `games/azul/` 负责 Azul 规则、状态、动作语义、Azul 专属 UI 与参数。
- 连接方式：
  - 训练：`azul/train/plugin.py` 注入 callbacks 到 `general/train`。
  - 调试：`azul/debug_service` 使用 `general/debug` 的通用服务壳。

## Azul 模块结构

### 1) 引擎（C++）

- 状态与规则：
  - `include/azul_state.h`
  - `include/azul_rules.h`
- `src/azul_state.cpp`
- `src/azul_rules.cpp`
- 搜索/网络适配：
- `src/azul_net_adapter.cpp`
- 调试扩展入口：
  - `debug_service/cpp_azul_engine_module.cpp`

### 2) 调试端（debug_service）

- `debug_service/setup.py`
  - 通过 `general.debug.cpp_extension_setup.build_cpp_extension(...)` 构建扩展。
- `debug_service/plugin.py`
  - 声明 `DEBUG_PLUGIN`，并通过通用工厂映射 Azul C++ 扩展到会话接口。
- `debug_service/app.py`
  - 通过 `general.debug.entrypoint.create_game_debug_app(...)` 装配服务。
- `debug_service/web/`
  - Azul 棋盘与交互前端实现。

### 3) 训练（train）

- `train/plugin.py`
  - 训练入口 + `TRAIN_PLUGIN` + selfplay/arena 导出 + MovePolicy/C++ 后端适配（当前已合并到同一文件）。
- `train/torch_trainer.py`
  - Azul 特化 trainer（稀疏 policy target + phase/value 权重）。
- `train/constants.py`
  - Azul 维度与标签常量。
- `train/train_config.example.json`
  - 默认训练配置单一真源。

## 现在的训练数据与产物

- 训练使用增量内存样本，不再写全量 `selfplay_samples.jsonl`。
- `selfplay_summary.json` 保留聚合信息，不再存全量 `results` 明细。
- 每个 step 不再额外落 `candidate_step_*.onnx/.pt` 快照。
- 保留核心产物：
  - `artifacts/models/candidate_model.onnx`
  - `artifacts/models/latest_model.onnx`
  - `artifacts/models/best_model.onnx`
  - `artifacts/train_summary.json`
  - `artifacts/selfplay_summary.json`
  - `artifacts/gating_summary.json`

## 以 Azul 为模板接入新游戏（落地清单）

### 引擎

- 实现 `<game>_state`、`<game>_rules`、`<game>_net_adapter`。
- 在 `debug_service/cpp_<game>_engine_module.cpp` 暴露：
  - 自博弈整局
  - arena 对战
  - 调试会话必需接口

### 调试前端

- 提供 `debug_service/web/` 游戏 UI。
- 提供 `debug_service/plugin.py`（推荐 `build_standard_game_debug_plugin(...)`）做通用接口绑定。
- `debug_service/app.py` 用 `create_game_debug_app(...)` 配置游戏参数。
- `debug_service/setup.py` 用通用 `build_cpp_extension(...)` 声明源文件。

### 训练

- 在 `train/plugin.py` 中提供训练入口、selfplay + arena 导出、MovePolicy 与 C++ 适配。
- 提供 `train/train_config.example.json`。
- 提供 `train/torch_trainer.py`：
  - 能复用通用骨架时，优先复用 `general.train.torch_simple_trainer`。
  - 仅保留本游戏特征抽取和标签策略。

