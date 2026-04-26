# TicTacToe 训练 MVP

本目录提供井字棋训练最小闭环：

1. C++ 整局 self-play
2. 样本落盘
3. Torch trainer（可导出 ONNX，未安装 torch 时自动跳过）
4. netmcts vs heuristic gating

## 运行

先在 `games/tictactoe/debug_service` 编译扩展并准备环境：

```powershell
cd "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/tictactoe/debug_service"
.\setup_env.ps1 -WithOnnx
```

再执行训练：

```powershell
..\..\..\.venv_debug_shared\Scripts\python -m games.tictactoe.train.plugin `
  --config "..\train\train_config.example.json" `
  --output "..\train\out\job_001"
```

## Torch 训练依赖（可选）

若要启用 trainer 导出 ONNX：

```powershell
..\..\..\.venv_debug_shared\Scripts\python -m pip install -r "..\..\..\requirements.train_torch.txt"
```

