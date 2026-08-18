[English](README.md) · **中文**

# GoldDash

GoldDash 是一个面向研究的贵金属信号系统，用于分析日线与盘中多时间尺度的黄金、白银市场。项目把市场与宏观特征、机器学习区间估计、波动率过滤、期权策略研究、回测和 Streamlit 仪表板整合到同一套可追溯流程中。

> **仅供研究使用。** GoldDash 是实验性研究代码，不是交易服务，也不构成投资建议。回测结果不等同于实盘表现，并会显著受到数据质量、交易成本、模型选择和市场状态变化的影响。

## 项目概览

| 方向 | 当前范围 |
|---|---|
| 研究品种 | 以 GLD、SLV、GC=F 和 SI=F 为主的黄金、白银研究 |
| 预测研究 | 基于 LSTM/Transformer 实验的多时间尺度区间估计 |
| 信号研究 | 趋势、已实现波动率、宏观、市场状态、盘中确认和期权衍生过滤 |
| 策略研究 | 期货、买入看涨期权、看跌信用价差、跨式和卖波动率结构 |
| 验证方式 | Walk-forward、多窗口回测、校准检查和分资产参数配置 |
| 使用界面 | Streamlit 仪表板与命令行研究脚本 |

## 研究流程

```text
市场与宏观数据
      ↓
特征构建与模型估计
      ↓
日级候选信号
      ↓
市场状态、趋势、波动率与盘中确认
      ↓
策略候选与风险参数
      ↓
回测 / 模拟持仓账本 / 仪表板
```

仓库将界面和研究逻辑分开：

- `core/`：数据读取、特征逻辑、模型、信号、校准和持仓账本。
- `scripts/`：数据准备、训练、回测、诊断和参数实验。
- `tests/`：离线单元测试与回归测试。
- `docs/`：架构、模型、策略和实验的详细说明。

## 核心能力

### 多源特征

GoldDash 组合价格、波动率、宏观、跨资产和市场结构信息。当前流程涵盖 GLD/SLV 与 COMEX 期货、已实现波动率、GVZ 等波动率指标、利率与美元因子、技术指标，以及部分期权和持仓量特征。

### 多时间尺度建模

项目包含日线和小时级建模实验。区间估计只是决策流程的输入之一，不被当作独立的价格预言。候选信号还需要经过趋势、波动率、市场状态、数据新鲜度和盘中确认。

### 策略与风险研究

策略层把过滤后的信号映射到分资产研究配置，分别评估仓位、持有期、退出规则和期权结构。这些模块均为模拟研究，不会自动下实盘订单。

### 可审计性

信号历史、校准记录、仪表板快照和持仓账本用于把界面上的结论追溯到数据和规则来源。

## 快速开始

### 1. 创建环境

```bash
git clone https://github.com/YuhaoDoong/BetaGold.git
cd BetaGold

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备数据

数据脚本会下载公开市场数据、可选地获取 FRED 序列、构建特征，并可训练区间模型：

```bash
# 只准备市场数据和特征
python scripts/setup_data.py --no-train

# 加入 FRED 数据；API key 可从 https://fred.stlouisfed.org/ 申请
python scripts/setup_data.py --fred-key YOUR_FRED_API_KEY --no-train

# 数据准备后单独训练
python scripts/setup_data.py --train-only
```

该脚本将数据写入仓库内的 `data/`。使用这一布局时，请把 `config.yaml` 中的 `data_root` 设置为 `"data"`。当前跟踪的配置仍反映维护者的本地研究环境，因此全新克隆后需要先修改该值。

### 3. 启动仪表板

```bash
streamlit run app.py
```

部分页面依赖未随仓库发布的模型文件、期权历史数据或模拟持仓账本。

## 验证与测试

只运行正式测试目录：

```bash
pytest -q tests
```

测试集同时包含完全独立的单元测试和依赖维护者历史市场快照的回归测试。全新克隆在相应本地数据尚未准备好时，可能出现缺少数据的失败。`pytest.ini` 已将 `scripts/` 下的实验脚本排除在自动测试发现之外。

研究结论应优先使用 walk-forward 和多窗口验证，而不是单一优化回测。常用入口包括：

```bash
python scripts/backtest_pipeline.py all
python scripts/monthly_retune.py --dry-run
python scripts/multi_window_validate.py
```

## 详细文档

- [系统架构](docs/ARCHITECTURE.md)：模块、数据流和部署方式
- [模型说明](docs/MODELS.md)：区间模型、校准和市场状态分类
- [策略说明](docs/STRATEGIES.md)：策略定义与风险假设
- [实验记录](docs/EXPERIMENTS.md)：研究过程与历史比较

## 数据与可复现性说明

- 公开数据源可能修改格式、修订历史值、限流或提供延迟数据。
- 模型文件、大型数据、私有券商导出和凭证不会提交到仓库。
- 部分旧研究脚本仍假设维护者的本地数据目录；在完成路径配置前，应将其视为实验记录。
- README 或文档中的胜率、收益只适用于对应样本、执行假设和成本设置，不能解释为未来预期回报。
- 任何真实资金使用都需要独立验证、真实滑点和费用建模、仓位限制及运行安全措施。

## 安全

不要提交凭证。`.gitignore` 已排除 `.env`、密钥文件和常见秘密文件名。券商凭证应通过环境变量或本地秘密管理工具提供。`setup_data.py` 可选保存的 FRED key 位于同样被忽略的 `.fred_key`。

## 项目状态

GoldDash 是持续迭代的个人研究项目。随着数据和验证协议变化，接口与实验脚本可能调整。欢迎通过 GitHub issue 提交可复现性反馈。
