# DinoBoard 新游戏实现教程（按当前架构）

本文是当前版本的**实操教程**。目标是：在不复制样板的前提下，把一个新游戏完整接入 DinoBoard。

核心原则：

- 游戏语义放 `games/<game>/...`
- 通用流程放 `general/...`
- 训练入口统一为 `games/<game>/train/plugin.py`
- 调试入口统一为 `games/<game>/debug_service/plugin.py`

---

## 0) 最小完成标准

完成接入后应满足：

- `python -m games.<game>.train.plugin --output ...` 可启动训练
- 能跑通 `selfplay -> (可选 train) -> gating`
- 调试服务可完成：开局、合法动作查询、人类落子、AI 落子
- `netmcts` 模式下模型路径真实生效（不是空转）
- 建议默认接入 `warm start` 路径（至少可配置为 `heuristic/random` 并可一键开启）

---

## 1) 先建最小文件集

训练侧（必须）：

- `games/<game>/train/plugin.py`（训练入口 + `TRAIN_PLUGIN` + selfplay runners）
- `games/<game>/train/train_config.example.json`

训练侧（按需）：

- `games/<game>/train/torch_trainer.py`
- `games/<game>/train/constants.py`
- `games/<game>/train/cpp_training_backend.py`（仅当你明确要拆分文件）

调试侧（必须）：

- `games/<game>/debug_service/plugin.py`（`DEBUG_PLUGIN`）
- `games/<game>/debug_service/app.py`（超薄模板）
- `games/<game>/debug_service/setup.py`（超薄模板）
- `games/<game>/debug_service/web/*`

C++ 侧（必须）：

- `games/<game>/include/*.h`
- `games/<game>/src/*.cpp`
- `games/<game>/debug_service/cpp_<game>_engine_module.cpp`

---

## 2) 实现 C++ 引擎（必须）

### 2.1 基础规则

在 `include/src` 完成：

- 状态表示
- 合法动作生成
- 动作推进与回滚（`do_action_fast / undo_action`）
- 终局结果（winner/shared_victory/scores）

要求：

- `ActionId` 编码稳定
- 回滚可用于搜索树反复展开
- 哈希稳定（用于搜索/复现）

### 2.2 NetMCTS 适配（推荐）

若支持 `netmcts`，实现 `<game>_net_adapter.h/.cpp`。

建议优先复用通用能力，不要在游戏侧重复实现：

- 根节点温度采样
- 根噪声窗口逻辑
- 通用 `search_options` 公共键解析

### 2.3 Python 扩展导出

`cpp_<game>_engine_module.cpp` 至少导出：

- `run_selfplay_episode_fast`
- `run_arena_match_fast`
- debug session 系列接口（`session_*`）

若支持 ONNX 推理，建议同时导出：

- `onnx_enabled`
- `feature_dim` / `action_space`（用于维度一致性校验）

---

## 3) 实现训练入口 `train/plugin.py`（必须）

### 3.1 声明 `TRAIN_PLUGIN`

必填字段：

- `description`
- `benchmark_engine`
- `support_benchmark_onnx`
- `move_policy_cls`
- `backend_factory`
- `game_type`
- `ruleset`
- `read_shared_victory_from_raw`

常见可选字段：

- `run_trainer`
- `build_initial_model`
- `pipeline_hooks`
- `before_run_job`
- `enrich_sample`
- `netmcts_data_source`
- `default_value_margin_weight`
- `default_value_margin_scale`

### 3.2 后端适配推荐写法（优先复用）

优先从 `general/train/cpp_training_backend.py` 选一种：

- `build_classic_backend_factory(...)`
- `build_search_options_backend_factory(...)`
- `build_flexible_search_options_backend_factory(...)`

若必须写自定义 backend，仍建议复用：

- `MovePolicy`
- `validate_netmcts_model_path(...)`

### 3.3 初始模型导出推荐写法

优先复用：

- `general/train/game_plugin.py::build_initial_model_from_exporter(...)`

这样可统一 `selfplay_init.onnx` 路径与参数包装，避免每个游戏重复写 `_build_initial_model`。

### 3.4 训练入口与 selfplay runners 导出（关键）

在 `plugin.py` 同文件导出：

- `run_job`
- `_default_config`
- `main`
- `run_selfplay_episode_payload`
- `run_arena_match`

推荐一行完成：

- `build_train_exports_from_current_game(__file__)`

并保留：

- `if __name__ == "__main__": raise SystemExit(main())`

### 3.5 训练目标一致性（强约束，避免策略塌缩）

若 C++ `run_selfplay_episode_fast` 已返回 MCTS 分布字段（如 `policy_action_ids` + `policy_probs` 或 `policy_action_visits`），
训练器**必须**消费这些稀疏分布做 policy 监督（例如 sparse target / KL / CE over distribution），
禁止仅用 `action_id` 做单标签 one-hot 监督。

原因：

- 只学最终选中动作会丢失搜索分布信息，容易在大动作空间下出现“重复动作/循环策略”塌缩。
- 截断局（`shared_victory=true`）样本多时，这个问题会被进一步放大。

最低验收要求：

- 抽样检查训练样本，确认 policy target 来自 MCTS 分布而非仅 `action_id`。
- 对局回放中不应长期出现“固定少数动作循环 + 高比例打满 `max_episode_plies`”。

### 3.6 Warm Start 接入（建议作为新游戏默认能力）

新游戏建议提供并验证 warm start：

- 至少支持 `warm_start_episodes > 0` 且 `warm_start_engine` 可切到 `heuristic`（或 `random`）。
- 对“几乎完全信息 + 低随机性”游戏，建议使用 `heuristic + 随机扰动` 启动数据（建议 `10%`，如 `heuristic_random_action_prob=0.1`），避免完全固定开局导致冷启动过慢。
- 建议保持可调：随机比例由 `search_options` 配置，可按游戏规模调整。

最低验收要求：

- 训练日志中可见 warm start 阶段执行（如 `warm start done`），并产出有效样本。
- 不应出现“warm start 全局面几乎固定、开局多样性极低”的明显退化现象。

### 3.7 ONNX 热更新与严格失败（强约束，避免“训练空转”）

若训练过程中会反复覆盖同一路径模型文件（例如 `candidate_model.onnx`），
**严禁** evaluator session 缓存仅按“模型路径”命中；缓存 key 必须包含文件指纹（至少 `size + mtime`）或显式版本号。

同时，`selfplay.policy.engine=netmcts` 时必须严格失败：

- C++ 扩展无法导入：直接报错并中止任务（不能静默降级）
- `onnx_enabled=false`：直接报错并中止任务

最低验收要求：

- 同一路径模型更新后，worker 侧推理 session 会被刷新（不是持续复用旧权重）。
- 人工制造“扩展不可导入/无 ONNX”场景时，训练应立即失败而不是继续跑。

### 3.8 完全信息无随机游戏的平局打破（强约束）

对“完全信息 + 无环境随机性”的游戏（如棋类/路径博弈），动作选择中的并列最优分支必须做**随机 tie-break**，
禁止固定返回“第一个动作/最小 action_id”。

要求：

- 适用位置至少包含：启发式对手并列最优动作、`temperature=0` 时访问数并列的根动作选择。
- tie-break 必须基于 `seed`（或可复现派生 seed）采样，保证“同 seed 可复现、不同 seed 可打散”。
- 不允许把动作容器顺序当作隐式策略（否则会产生系统性先手/方位偏置）。

最低验收要求：

- `policy_a` 对 `policy_a` 自对弈（交换先后手）胜率不应长期固定在机械模式（如恒定 0/1/0.5）。
- 改变 seed 后，开局若存在并列最优动作，首步动作分布应出现可观察差异。

---

## 4) 实现调试服务（必须）

交互原则（强约束）：

- 调试前端必须渲染游戏画面（棋盘/桌面/卡池等），并在该画面上直接交互执行动作。
- 禁止把“每个合法动作”都做成固定按钮或下拉列表让用户逐条选 action_id。
- 人类操作应尽量映射为游戏语义操作（如点格走子、点边放墙、点卡购买），由前端转换为对应 action。

### 4.1 `debug_service/plugin.py`

声明 `DEBUG_PLUGIN`，推荐复用：

- `general/debug/game_plugin.py::build_standard_game_debug_plugin(...)`

### 4.2 `debug_service/app.py`（超薄模板）

推荐只保留：

- `create_debug_app_from_current_game(__file__)`

对应实现位于：

- `general/debug/plugin_loader.py`

### 4.3 `debug_service/setup.py`（超薄模板）

推荐只保留：

- `build_cpp_setup_kwargs_from_current_game(...)`

说明：通用 `setup_game_env.ps1` 会优先从 `DEBUG_PLUGIN.cpp_extension_name` 自动解析 C++ 模块名。

---

## 5) 脚本与配置约定（当前标准）

- 默认训练配置单一真源：`games/<game>/train/train_config.example.json`
- 训练启动统一脚本：`general/scripts/run_game_train.ps1`
  - 内部执行 `python -m games.<game>.train.plugin`
- 调试环境统一脚本：`general/scripts/setup_game_env.ps1`
- 调试启动统一脚本：`general/scripts/run_game_debug.ps1`
- 每个游戏自身 `setup_env.ps1 / run_debug.ps1` 仅保留薄包装

---

## 6) 验收步骤（建议照抄）

1. 编译并安装 debug 扩展（含依赖）
2. 启动调试服务，完成人机基础操作
3. 运行训练入口：
   - `python -m games.<game>.train.plugin --output ...`
4. 检查核心产物：
   - `selfplay_summary.json`
   - `train_summary.json`
   - `gating_summary.json`
   - `artifacts/models/{candidate,best,latest}`
5. 在 `netmcts` 下做有/无模型路径对比，确认行为变化

---

## 7) 常见错误（高频）

- 在游戏目录重复造 CLI 壳、selfplay 壳、setup 壳
- 在前端或插件里硬编码机器绝对路径
- `TRAIN_PLUGIN` / `DEBUG_PLUGIN` 字段多处定义，失去单一真源
- 把通用逻辑塞到 `games/<game>`，导致后续游戏继续复制
- 继续使用旧入口 `run_training_job.py`（当前标准入口是 `train/plugin.py`）
- C++ 已产出 MCTS 分布，但训练仍只用 `action_id` 单标签监督，导致策略循环/和棋塌缩
- 新游戏未接入 warm start，或 warm start 纯确定性（无扰动）导致冷启动慢/开局塌缩

---

## 8) 一句话模板

新游戏接入时优先做到：

- C++ 规则和扩展是游戏特化
- `train/plugin.py` 与 `debug_service/plugin.py` 是唯一入口
- 其他流程尽量调用 `general` 现成工厂/loader，不复制样板

