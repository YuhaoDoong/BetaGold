"""Test individual kline requests to diagnose failures"""
from moomoo import OpenQuoteContext, RET_OK

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

tests = [
    ('US.GLD260618C276000', 'Jun26 C276 (known working)'),
    ('US.GLD260618C450000', 'Jun26 C450 (high strike)'),
    ('US.GLD270115C450000', 'Jan27 C450 (LEAPS)'),
    ('US.GLD260325C466000', 'Mar25 C466 (weekly)'),
    ('US.GLD260417C400000', 'Apr17 C400'),
    ('US.GLD261218C400000', 'Dec26 C400'),
]

for code, desc in tests:
    ret, data, _ = ctx.request_history_kline(
        code, start='2025-01-01', end='2026-03-11')
    if ret == RET_OK:
        print(f"OK   {desc}: {len(data)} rows")
        if not data.empty:
            print(f"     {data['time_key'].iloc[0][:10]} ~ "
                  f"{data['time_key'].iloc[-1][:10]}")
    else:
        print(f"FAIL {desc}: {str(data)[:100]}")

ctx.close()
