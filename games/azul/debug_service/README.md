# DinoBoard Intelligence Azul Debug Service（可验收 Demo）

通用环境/启动流程见：`general/README.md` 的 “Debug Service 通用约定”。

这个服务是当前阶段的可验收成果，目标是先打通（且已切到 C++ 引擎）：

- `Azul` 规则推进（C++ `AzulState/AzulRules`）
- 对局 API
- 回放事件/帧
- `step-back`
- `force-opponent-move`
- 网页调试端

## 1. 项目共享调试环境（全游戏共用）

首次初始化（PowerShell）：

```powershell
cd "DinoBoard Intelligence/games/azul/debug_service"
.\setup_env.ps1
.\run_debug.ps1
```

日常启动（不重复安装依赖）：

```powershell
cd "DinoBoard Intelligence/games/azul/debug_service"
.\run_debug.ps1
```

强制重装依赖：

```powershell
cd "DinoBoard Intelligence/games/azul/debug_service"
.\setup_env.ps1 -ForceInstall
```

手动命令（可选）：

```powershell
cd "DinoBoard Intelligence/games/azul/debug_service"
..\..\..\.venv_debug_shared\Scripts\python -m pip install -r ..\..\..\requirements.debug_shared.txt
..\..\..\.venv_debug_shared\Scripts\python -m pip install -e .
..\..\..\.venv_debug_shared\Scripts\python -m uvicorn app:app --reload --app-dir "." --port 8000
```

浏览器打开：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 2. 已实现接口

- `POST /api/v1/games`
- `GET /api/v1/games/{session_id}/state`
- `GET /api/v1/games/{session_id}/legal-actions`
- `POST /api/v1/games/{session_id}/actions`
- `POST /api/v1/games/{session_id}/ai-move`（支持 `heuristic` / `netmcts`）
- `POST /api/v1/games/{session_id}/force-opponent-move`
- `POST /api/v1/games/{session_id}/step-back`
- `GET /api/v1/games/{session_id}/replay`
- `GET /api/v1/games/{session_id}/frames`
- `GET /api/v1/games/{session_id}/frames/{ply}`
- `POST /api/v1/games/{session_id}/seek`
- `POST /api/v1/games/{session_id}/rebuild-frames`

## 3. 验收建议

1. 开局后点击 legal action 按钮走几步。  
2. 点击 `AI 走一步`，确认状态版本递增。  
3. 点击 `悔棋`，确认状态回退且 legal actions 变化。  
4. 在对手回合尝试 `force-opponent-move`（可用接口直接测）。  
5. 点击 `查看 replay`，确认事件序列和 ply 连续。  

## 4. 当前限制

- `netmcts` 已接入 C++ 通用 MCTS（`general/search/net_mcts.*`）+ ONNX evaluator 管线。
- 默认构建不启用 ONNX Runtime（会退化为均匀先验 + 0 value，便于无依赖联调）。
- 启用 ONNX Runtime 时设置环境变量后重装：
  - `.\setup_env.ps1 -ForceInstall -WithOnnx -OnnxRuntimeRoot "<onnxruntime 安装根目录>"`
- 网页是调试版（偏数据检查），不是最终游戏化 UI。
- 会话数据当前在内存中，服务重启后会丢失。

## 5. 前端分区（本轮拆分）

- `web/app.js`：页面编排与事件绑定（轻量入口）
- `web/core/state.js`：会话状态与常量
- `web/core/dom.js`：DOM 读写小工具
- `web/core/utils.js`：Azul 前端通用计算/样式工具
- `web/controllers/move_flow.js`：走子提交与动画流程
- `general/web/two_player_turn_runtime.js`：开局/悔棋/替对手落子/AI响应等通用回合控制
- `web/render/factories_center.js`：工厂区与中央池渲染
- `web/render/player_board.js`：玩家板与地板渲染
- `web/render/selection_panel.js`：选中来源/操作提示区渲染
- `web/styles.css`：样式入口（聚合）
- `web/styles/layout.css`：布局与侧栏
- `web/styles/table.css`：桌面区与工厂/中央池
- `web/styles/player.css`：玩家板/墙面/地板
- `web/styles/effects.css`：动画、工具类与响应式

原则：

- 可通用能力继续放在 `general/web/*`。
- Azul 语义渲染保留在 `games/azul/debug_service/web/*`。
- 单文件超 500 行时优先拆分后再继续加功能。
- 侧边栏交互（开局/难度/开始/悔棋/替对手落子）通过 `general/web/sidebar_controls.js` 统一绑定。

## 6. 后端分区（本轮拆分）

- `general/debug/service_interfaces.py`：通用后端会话接口定义。
- `general/debug/runtime_service.py`：通用会话编排（state_version、错误码、回放/悔棋流程）。
- `games/azul/debug_service/plugin.py`：Azul 的调试插件配置（委托通用 C++ backend 工厂）。
- `games/azul/debug_service/app.py`：FastAPI 路由壳层。
