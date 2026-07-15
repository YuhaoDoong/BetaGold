# Version 3: 策略引擎层

将预测信号转化为具体期权交易决策。

## 模块

| 模块 | 功能 |
|------|------|
| signal_fusion.py | 融合宏观/价格/波动率三个预测模块的信号 |
| options_strategy.py | 策略选择器: 信号组合 -> 具体期权策略 |
| risk_management.py | 仓位控制、Greeks暴露限制、止损止盈 |

## V1 策略范围

1. 单腿方向单 (买 Call / 买 Put)
2. 垂直价差 (Bull Call Spread / Bear Put Spread / Credit Spread)

跨式/宽跨式放到 V2。

## 状态

待开发 (Version 3)
