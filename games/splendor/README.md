# DinoBoard Intelligence - Splendor 模块

`games/splendor/` 是璀璨宝石（2 人）在 DinoBoard 通用训练框架里的独立落地目录。

当前版本目标是先完成**C++ 引擎 + 训练闭环**：

- 按 `general/train` 的统一接口接入新游戏；
- C++ 实现 Splendor 核心规则（买牌/预购/拿宝石/贵族结算/终局判定）；
- 开发卡使用官方完整 90 张配置逐条硬编码，贵族使用官方 10 张表；
- 超宝石上限通过显式动作空间处理（返还动作），不做自动弃牌；
- 搜索状态使用**可持久化结构**（父节点共享 + 惰性物化），避免博弈树前进/回退的重复整状态拷贝；
- 支持 `heuristic` / `netmcts` 统一入口；
- 训练侧复用 `general.train.torch_simple_trainer`，仅保留本游戏特征与动作空间。

## 目录

- `include/` + `src/`：Splendor C++ 状态/规则/NetAdapter
- `debug_service/cpp_splendor_engine_module.cpp`：Python 扩展入口（含 selfplay/arena/session）
- `debug_service/setup.py`：C++ 扩展构建配置
- `train/plugin.py`：训练入口（对接通用 pipeline） + selfplay/arena 回调 + `TRAIN_PLUGIN` + 训练后端适配（调用 `cpp_splendor_engine_v1`）
- `train/torch_trainer.py`：特征抽取 + 通用 trainer 复用
- `train/train_config.example.json`：默认训练配置单一真源

## 说明

- 为避免重复实现通用编排逻辑，本模块不修改 `general/`；
- 为避免污染参考仓库，本模块不改动 `mosaic-azul` 与 `alpha-zero-general-master`；
- 当前训练/调试主路径均走 C++ 引擎。

