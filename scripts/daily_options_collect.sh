#!/bin/bash
# 每日自动采集 GLD 期权链快照
# 美股收盘后运行 (4pm ET = 北京时间次日 4~5am)

cd /Users/yhdong/Gold

# v3.7.199: cron 没有 conda PATH, 直接用绝对 python 路径
# 旧 'conda run -n gold python ...' 自 5/6 起每日失败 → kline_db 停在 5/6
PYTHON="/Users/yhdong/miniconda3/envs/gold/bin/python"
OPEND="/Users/yhdong/Software/moomoo_OpenD_9.6.5618_Mac/moomoo_OpenD_9.6.5618_Mac/OpenD.app/Contents/MacOS/OpenD"
LOG="/Users/yhdong/Gold/logs/options_collect.log"
mkdir -p /Users/yhdong/Gold/logs

echo "$(date '+%Y-%m-%d %H:%M:%S') === 开始采集 ===" >> "$LOG"

# 检查 OpenD 是否运行，未运行则启动
if ! pgrep -f "OpenD.app/Contents/MacOS/OpenD" > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 启动 OpenD..." >> "$LOG"
    "$OPEND" &>/dev/null &
    sleep 10  # 等待 OpenD 初始化
fi

# 采集 EOD 全链快照
"$PYTHON" -m src.data.options.options_archive >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') === 采集完成 ===" >> "$LOG"
echo "" >> "$LOG"
