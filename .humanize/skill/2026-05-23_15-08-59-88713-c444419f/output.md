AGREE:
- v1 已正确修正 Calibration label：以 `build_targets()` 的 5-day forward H/L vs t-day close 为准，AC-1 的正反测试方向 sound。
- `get_option_exit_config(asset, strategy)` 单一 registry 是合理方向，优于把 exit threshold 混进 `AssetConfig`。
- STRADDLE / SHORT_VOL expiry intrinsic 扩展是必须项，且要求策略特定 payoff 不复用 BC/SP helper，这点方向正确。
- Layer 2 改成 per-leg DTE sample disposition 是正确修正，能避免 closed-only survivorship bias。
- Dashboard `run_backtest` 保留 compat wrapper + parity assertion，比 one-shot replacement 更安全。
- `PENDING_KLINE` 与 `NO_CONTRACT` 分离、kline stale 只 gate 新 option entry、不阻断 futures / MTM / expiry intrinsic，这个边界合理。
- Calibration shadow-first + Layer1 scoreB gate 后再 live cutover 是必要的，避免把“统计校准”直接变成生产交易行为变化。
- 加 `pytest`、offline fixtures、per-tag `VALIDATION.md`，符合 correctness floor 的验证需求。

DISAGREE:
- AC-8 的 “shift(1) discipline” 对 5-day forward label 不够。`actual_upper/lower` 对 t 日预测要等 t+5 后才成熟；在 t 日校准时使用 t-1 的 actual 仍然 look-ahead。需要按 label horizon 做 maturity lag，而不是只 shift(1)。
- SHORT_VOL IC max_risk 写成 `min(call_wing_width, put_wing_width) - credit` 不 sound。非对称 wing 时最大风险应按更宽一侧，即 `max(...) - credit`，除非代码强制 symmetric wings 并有 assertion。
- Lower Bound 允许 “AC-7 report-only enhancement without changing sample-restriction logic”，这和 AC-7 正文要求 per-leg DTE filter 直接冲突。
- 原始 draft appendix 仍保留 5-6× / 87.6% / build_band 自动读 calibrated 的旧说法；虽然叫 Original Draft，但 plan 要求 implementer 读全文件，旧口径会污染执行。
- Goal 声称消除 “5 severe + 6 medium + 4 optimization defects”，但主 AC/task 没覆盖 OI 修正进主链、risk controls 迁入主链、yfinance bid/ask mid fallback。要么补任务，要么缩窄 Goal。
- SP fallback sign、`entry_spot` 语义迁移、Dashboard parity 都在 task 里，但没有独立 AC；现在 task-a2/a3/e3 挂到 AC-3/AC-5 不够精确。
- Calibration scaler 仍隐含“bands 应缩窄”的旧假设。当前事实是 mean width 偏宽但 coverage 低，说明问题可能是分布形状/尾部/方向错配；硬性要求 calibrated narrower 可能进一步降低 coverage。

REQUIRED_CHANGES:
- 把 AC-8 / task-g3 / task-f4 改成 label-maturity-safe：5-day forward actual 只能在 window 完成后进入 residual pool，例如 `label_end_date < calibration_as_of_date` 或显式 `horizon=5` lag。`apply_rolling_conformal_scaler` 的签名也应能表达 dates/horizon，或明说输入 actual 已按 maturity lag 预处理。
- 修正 SHORT_VOL IC max_risk：用 `max(call_wing_width, put_wing_width) - credit`，并测试 asymmetric wing fixture。
- 删除或明确标注 Original Design Draft 为 “non-normative / superseded”；尤其是旧 calibration 数字、v3.7.241 简化 sample filter、v3.7.246 自动 fallback 读 calibrated、v3.7.249 “calibrated narrower” 测试。
- 统一 Lower Bound 与 AC：如果 AC-7 是接受标准，Lower Bound 不能允许跳过 per-leg DTE sample restriction。
- 增补 AC 覆盖遗漏行为：SP fallback sign、`underlying_entry_price`/`entry_spot` migration、Dashboard `run_backtest` parity、OI 主链修正、risk controls、bid/ask mid fallback；或把 Goal 改成只覆盖本轮列出的 correctness subset。
- 定义 expiry-day 行为：`today == expiry_dt` 且 kline missing 时是否允许 intrinsic close，取决于 underlying expiry close 是否已知。现在 AC-4 的 12/16 场景表述不够判定实现对错。
- 明确 cross-asset selector 对 `gvz_value` missing/stale 的策略：skip、BUY CALL fallback、还是 block entry；否则 IV-aware rule 在真实数据缺口下不可验证。
- 修正 AC-5 “shadow-only flag default off records...” 的语义：建议区分 `live_cutover=False` 与 `shadow_logging=True`，否则默认 off 无法积累 14 天 gate 证据。

OPTIONAL_IMPROVEMENTS:
- 把 calibration 的目标拆成两个指标：coverage calibration 与 trading scoreB gate。不要把“缩窄 band”当成测试目标。
- AC-10 的 bit-identical validation 建议指定 timestamp/random seed/output-normalization，否则 grid archive 很容易因运行时间字段不一致失败。
- `core/calibration.py` runtime assertion 读取 `gate_report.md` 可能让生产 import 依赖文件系统状态；更稳的是 config validation 或 preflight check。
- DEC-1/DEC-4 已在正文多处倾向新 registry / wrapper-retention；如果实际上已经决定，应从 Pending Decision 移到 Resolved Decision，减少 implementer 分叉。

UNRESOLVED:
- 我与 v1 的主要对立点是 calibration maturity lag：我认为 shift(1) 不足以防 5-day forward label 泄漏，必须改为 horizon-aware maturity gate。
- IC max_risk 的 `min` vs `max` 需要用户确认。如果策略永远 symmetric wings，可保留但必须加 symmetric assertion；否则必须用 `max`。
- Scope 需要用户决定：本 plan 是否真的要覆盖所有 5 severe + 6 medium + 4 optimization，还是只覆盖当前 AC 列出的 correctness/calibration floor。
