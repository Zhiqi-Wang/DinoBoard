# Quoridor 训练

本目录是 Quoridor 的训练入口与参数样例：

- 入口：`games/quoridor/train/plugin.py`
- 默认配置：`games/quoridor/train/train_config.example.json`

先在 `games/quoridor/debug_service` 编译扩展并准备环境，再启动训练：

```powershell
cd "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/quoridor/debug_service"
.\setup_env.ps1 -WithOnnx

cd "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence"
.\.venv_debug_shared\Scripts\python -m games.quoridor.train.plugin --config "games/quoridor/train/train_config.example.json" --output "games/quoridor/train/out/job_001"
```

## 重要：特征视角版本

`quoridor` 的网络输入采用“当前执子方视角归一（对手视角会做 180 度翻转）”。

当该编码规则变更后，旧 ONNX/PT 模型不再可直接复用。请重新训练并重新导出模型，
不要把旧 run 的 `candidate/best` 继续用于新版本评估。

