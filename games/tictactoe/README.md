# DinoBoard Intelligence - TicTacToe 模块

`tictactoe/` 是独立新游戏落地目录，包含：

- C++ 引擎（状态/规则/MCTS）
- 调试服务（FastAPI + 简易网页）
- 训练 MVP（复用 `general/train/pipeline.py`）

约束：

- 不修改 `general/` 与 `azul/` 既有实现
- 新能力优先在本目录完成验证

