"""测试 Moomoo API 期权历史K线能回溯多远"""
from moomoo import OpenQuoteContext, RET_OK
import pandas as pd
import time

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret, exp_df = ctx.get_option_expiration_date('US.GLD')
exp_df['strike_time'] = pd.to_datetime(exp_df['strike_time'])

# 选几个不同远近的到期日测试
targets = [7, 30, 90, 180, 365, 700]
for min_dte in targets:
    subset = exp_df[exp_df['option_expiry_date_distance'] >= min_dte]
    if subset.empty:
        continue
    row = subset.iloc[0]
    exp = row['strike_time'].strftime('%Y-%m-%d')
    dte = int(row['option_expiry_date_distance'])

    ret2, chain = ctx.get_option_chain('US.GLD', start=exp, end=exp)
    if ret2 != RET_OK:
        continue

    # 取ATM附近的call (GLD ~276)
    calls = chain[chain['option_type'] == 'CALL'].copy()
    calls['strike'] = calls['code'].str.extract(r'C(\d+)').astype(float) / 1000
    atm_idx = (calls['strike'] - 276).abs().argsort().iloc[0]
    atm = calls.iloc[atm_idx]
    code = atm['code']

    # 获取全部历史
    ret3, kline, _ = ctx.request_history_kline(
        code, start='2024-01-01', end='2026-03-11')
    if ret3 == RET_OK and not kline.empty:
        print(f'到期{exp} (DTE={dte:4d}d) | {code}')
        print(f'  K线: {len(kline):3d} 天, '
              f'{kline["time_key"].iloc[0][:10]} ~ {kline["time_key"].iloc[-1][:10]}')
    else:
        print(f'到期{exp} (DTE={dte:4d}d) | 无数据或失败')

    time.sleep(1)

ctx.close()
