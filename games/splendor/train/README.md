# Splendor 训练说明（DinoBoard Intelligence）

本目录是 Splendor 的训练接入层，复用 `general/train` 主流程。

## 当前能力

1. C++ 引擎整局 self-play（`cpp_splendor_engine_v1`）
2. 训练入口 hooks（分差 margin + phase 标签）
3. 增量样本进入 replay buffer（保留稀疏策略分布）
4. Torch policy/value 训练并导出 ONNX（未装 torch 时自动跳过）
5. return 阶段样本权重下调（不丢弃样本）
6. 显式统计 `return` / `choose_noble` 子阶段样本占比
7. `netmcts` 自博弈（strict mode：ONNX 不可用会直接报错）
8. 训练期 `eval/gating` benchmark 固定为 `heuristic`
9. 自动导出 `selfplay_init.onnx` 作为冷启动模型
10. 训练前 C++/Python 维度一致性护栏（feature/action space）

## 关键配置说明

- `train_config.example.json` 默认 `selfplay.policy.engine=netmcts`，`warm_start_engine=heuristic`。
- 训练入口启用严格模式：当 `engine=netmcts` 但扩展未启用 ONNX 时会直接报错，不会自动回退到 `mcts`。
- 训练期 benchmark 引擎固定为 `heuristic`（用于 eval/gating 对手）。
- Splendor 的随机拿牌在 netmcts 中按 chance 处理：抽牌转移会进行随机化并继续展开，不再在随机节点直接截断到叶子估值；可通过 `search_options.stop_on_draw_transition / enable_draw_chance / chance_expand_cap` 控制。
- `trainer.return_phase_factor` 控制 return 子阶段价值损失权重（0~1，默认 0.6）。
- `ChooseNoble` 阶段目前已进入样本统计，但暂不单独做 value loss 降权。
- `trainer.value_late_weight` 结合样本 `phase` 做后期样本增权。
- 当前默认训练配置（`train_config.example.json`）：
  - `selfplay.episodes=50`
  - `selfplay.warm_start_episodes=10000`
  - `selfplay.warm_start_train_passes=2`
  - `selfplay.policy.simulations=500`
  - `selfplay.mcts_schedule.start/end=10/500`
  - `selfplay.policy.temperature=0.8`（固定温度）
  - `search_options.stop_on_draw_transition=true`
  - `trainer.batch_size=2048, hidden=512, mlp_layers=4, updates_per_step=6, buffer_size=300000`
  - `trainer.value_late_weight=0.5, trainer.return_phase_factor=0.6`
  - `gating.games=100, gating.eval_every_steps=100, gating.eval_games=100, gating.history_best_games=100`

## 运行方式

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/splendor/debug_service/setup_env.ps1" -WithOnnx
```

然后执行训练：

```powershell
python -m games.splendor.train.plugin `
  --config "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/splendor/train/train_config.example.json" `
  --output "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/splendor/train/out/job_001"
```

或仅指定输出目录（自动读取 `train_config.example.json`）：

```powershell
python -m games.splendor.train.plugin `
  --output "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/splendor/train/out/job_default"
```

## 单测（可见性一致性）

```powershell
python -m unittest "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/splendor/train/test_reserved_visibility_consistency.py"
```


