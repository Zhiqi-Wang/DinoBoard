# TicTacToe 调试服务

通用环境/启动流程见：`general/README.md` 的 “Debug Service 通用约定”。

## 一次性初始化

```powershell
cd "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/tictactoe/debug_service"
.\setup_env.ps1 -WithOnnx
```

说明：脚本会使用项目根目录共享环境 `DinoBoard Intelligence/.venv_debug_shared`，并编译 `C++ netmcts + onnx` 推理链路。

## 启动

```powershell
cd "d:/cursor/Board_game_AI_framework/DinoBoard Intelligence/games/tictactoe/debug_service"
.\run_debug.ps1 -Port 8011
```

浏览器打开：

- [http://127.0.0.1:8011/](http://127.0.0.1:8011/)

## 当前支持

- 新局 / 落子 / AI 走子 / 悔棋
- replay 与 frames 接口
- C++ 全局 self-play 与 arena（训练调用）

