"""
深度学习波动区间预测 (DL Range V2)

预测未来5天的价格波动区间:
    upper = max(High_t+1, ..., High_t+5) / Close_t - 1
    lower = min(Low_t+1, ..., Low_t+5) / Close_t - 1

模型: LSTM + Attention + Quantile Loss
训练: RV归一化 + 独立Conformal校准 + 多种子集成

用法:
    from core.dl_range import DLRangePredictor, SELECTED_FEATURES
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


# ======================================================================
# Feature Selection
# ======================================================================

SELECTED_FEATURES = [
    # 价格动量 (多时间尺度)
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # 技术指标
    "rsi_14", "macd_hist", "bb_position", "stoch_k_14",
    "close_to_sma_5", "close_to_sma_20", "close_to_sma_60", "close_to_sma_120",
    "sma_20_slope", "sma_60_slope", "ma_alignment",
    "atr_14_pct", "daily_range_pct",
    # 宏观因子
    "real_yield_10y", "real_yield_10y_change_20d",
    "tw_usd", "tw_usd_ret_20d", "dxy_ret_5d",
    "fed_funds_rate", "fed_funds_rate_change_60d",
    "breakeven_10y", "us10y_level", "us10y_change_5d",
    "cpi_yoy", "m2_yoy",
    # 波动率
    "gvz", "gvz_pctile_252d",
    "vix_level", "vix_term_slope",
    "rv_20d", "hv_60d",
    "iv_rv_spread", "vrp_20d",
    # 持仓/资金流
    "cot_noncomm_net_change", "cot_noncomm_net_pctile",
    "cot_oi_change_pct",
    "cb_global_12m_rolling",
    # 跨市场
    "copper_gold_ratio_change", "gold_silver_ratio",
    "gc_gld_ratio_zscore",
]


def select_features(df: pd.DataFrame) -> list:
    """从 DataFrame 中选择可用的特征列."""
    return [f for f in SELECTED_FEATURES if f in df.columns]


# ======================================================================
# Dataset
# ======================================================================

class RangeDataset(Dataset):
    """时序数据集: (seq_len, n_features) → (upper_norm, lower_norm)"""

    def __init__(self, features: np.ndarray,
                 upper_targets: np.ndarray,
                 lower_targets: np.ndarray,
                 seq_len: int = 20):
        self.features = features.astype(np.float32)
        self.upper = upper_targets.astype(np.float32)
        self.lower = lower_targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.upper) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        u = self.upper[idx + self.seq_len - 1]
        l = self.lower[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(u), torch.tensor(l)


# ======================================================================
# Loss
# ======================================================================

class QuantileLoss(nn.Module):
    """Pinball / Quantile Loss."""

    def __init__(self, q_upper: float = 0.85, q_lower: float = 0.15):
        super().__init__()
        self.q_upper = q_upper
        self.q_lower = q_lower

    def forward(self, pred_upper, pred_lower, actual_upper, actual_lower):
        err_u = actual_upper - pred_upper
        loss_u = torch.where(
            err_u > 0,
            self.q_upper * err_u,
            (1 - self.q_upper) * (-err_u)
        ).mean()

        err_l = actual_lower - pred_lower
        loss_l = torch.where(
            err_l < 0,
            (1 - self.q_lower) * (-err_l),
            self.q_lower * err_l
        ).mean()

        return loss_u + loss_l


# ======================================================================
# Model
# ======================================================================

class RangeLSTM(nn.Module):
    """LSTM + Attention 预测波动区间 sigma multiplier."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attn = nn.Linear(hidden_size, 1)

        self.upper_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.lower_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch, seq_len, feat = x.shape
        x_flat = x.reshape(-1, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        lstm_out, _ = self.lstm(x)

        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)

        upper = self.upper_head(context).squeeze(-1)
        lower = self.lower_head(context).squeeze(-1)
        return upper, lower


# ======================================================================
# Predictor (high-level API)
# ======================================================================

class DLRangePredictor:
    """
    深度学习波动区间预测器 (V2: 归一化 + 独立校准 + 集成).

    训练流程:
        1. 目标归一化: upper/lower 除以 rv_scale
        2. 在 train 上训练, val 上 early stopping
        3. 在独立 cal 集上做 conformal calibration
        4. n_ensemble 个模型取平均
    """

    def __init__(self, seq_len: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 150, batch_size: int = 64,
                 patience: int = 20, device: str = "auto",
                 q_upper: float = 0.85, q_lower: float = 0.15,
                 n_ensemble: int = 3,
                 cal_target_cov: float = 0.80):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.q_upper = q_upper
        self.q_lower = q_lower
        self.n_ensemble = n_ensemble
        self.cal_target_cov = cal_target_cov

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.scaler = RobustScaler()
        self.models = []
        self.n_features = None
        self.cal_upper_margin = 0.0
        self.cal_lower_margin = 0.0

    def fit(self, features, upper_targets, lower_targets,
            rv_scale,
            val_features=None, val_upper=None, val_lower=None,
            val_rv_scale=None,
            cal_features=None, cal_upper=None, cal_lower=None,
            cal_rv_scale=None,
            verbose=False):
        """训练集成模型. cal_* 参数: 独立校准集."""
        self.n_features = features.shape[1]

        X_scaled = self.scaler.fit_transform(features)
        X_scaled = np.nan_to_num(X_scaled, 0)

        rv_safe = np.clip(rv_scale, 0.5, None)
        upper_norm = upper_targets / rv_safe
        lower_norm = lower_targets / rv_safe

        dataset = RangeDataset(X_scaled, upper_norm, lower_norm, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True)

        val_loader = None
        if val_features is not None:
            X_val = self.scaler.transform(val_features)
            X_val = np.nan_to_num(X_val, 0)
            rv_val_safe = np.clip(val_rv_scale, 0.5, None)
            val_dataset = RangeDataset(
                X_val, val_upper / rv_val_safe, val_lower / rv_val_safe,
                self.seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False)

        self.models = []
        for seed_i in range(self.n_ensemble):
            torch.manual_seed(42 + seed_i * 7)
            np.random.seed(42 + seed_i * 7)

            model = RangeLSTM(
                self.n_features, self.hidden_size,
                self.num_layers, self.dropout
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5)
            criterion = QuantileLoss(self.q_upper, self.q_lower)

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0

            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0
                n_batches = 0
                for X_batch, u_batch, l_batch in loader:
                    X_batch = X_batch.to(self.device)
                    u_batch = u_batch.to(self.device)
                    l_batch = l_batch.to(self.device)

                    optimizer.zero_grad()
                    pred_u, pred_l = model(X_batch)
                    loss = criterion(pred_u, pred_l, u_batch, l_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train = epoch_loss / max(n_batches, 1)

                val_loss = avg_train
                if val_loader is not None:
                    val_loss = self._eval_loss(model, val_loader, criterion)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(self.device)

            self.models.append(model)

        if cal_features is not None:
            self._calibrate(cal_features, cal_upper, cal_lower, cal_rv_scale)

        return self

    def _calibrate(self, cal_features, cal_upper, cal_lower, cal_rv_scale):
        """Conformal calibration on independent calibration set."""
        pred_u, pred_l = self._predict_raw(cal_features, cal_rv_scale)
        n_pred = len(pred_u)

        actual_u = cal_upper[self.seq_len - 1:][:n_pred]
        actual_l = cal_lower[self.seq_len - 1:][:n_pred]

        valid = ~(np.isnan(actual_u) | np.isnan(actual_l))
        actual_u, actual_l = actual_u[valid], actual_l[valid]
        pred_u, pred_l = pred_u[valid], pred_l[valid]

        upper_residual = actual_u - pred_u
        lower_residual = pred_l - actual_l

        per_side_target = np.sqrt(self.cal_target_cov)
        q_pct = per_side_target * 100

        self.cal_upper_margin = float(np.percentile(upper_residual, q_pct))
        self.cal_lower_margin = float(np.percentile(lower_residual, q_pct))

        self.cal_upper_margin = max(self.cal_upper_margin, -0.5)
        self.cal_lower_margin = max(self.cal_lower_margin, -0.5)

    def _predict_raw(self, features, rv_scale):
        """集成预测 (归一化空间 → 乘回 rv_scale), 不加 margin."""
        X_scaled = self.scaler.transform(features)
        X_scaled = np.nan_to_num(X_scaled, 0)

        dataset = RangeDataset(
            X_scaled,
            np.zeros(len(X_scaled)),
            np.zeros(len(X_scaled)),
            self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_uppers = []
        all_lowers = []
        for model in self.models:
            model.eval()
            uppers, lowers = [], []
            with torch.no_grad():
                for X_batch, _, _ in loader:
                    X_batch = X_batch.to(self.device)
                    pred_u, pred_l = model(X_batch)
                    uppers.append(pred_u.cpu().numpy())
                    lowers.append(pred_l.cpu().numpy())
            all_uppers.append(np.concatenate(uppers))
            all_lowers.append(np.concatenate(lowers))

        avg_upper_norm = np.mean(all_uppers, axis=0)
        avg_lower_norm = np.mean(all_lowers, axis=0)

        rv_aligned = rv_scale[self.seq_len - 1:]
        rv_safe = np.clip(rv_aligned, 0.5, None)

        return avg_upper_norm * rv_safe, avg_lower_norm * rv_safe

    def predict(self, features, rv_scale):
        """预测并应用 conformal margin."""
        pred_u, pred_l = self._predict_raw(features, rv_scale)
        pred_u = pred_u + self.cal_upper_margin
        pred_l = pred_l - self.cal_lower_margin
        return pred_u, pred_l

    def _eval_loss(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for X_batch, u_batch, l_batch in loader:
                X_batch = X_batch.to(self.device)
                u_batch = u_batch.to(self.device)
                l_batch = l_batch.to(self.device)
                pred_u, pred_l = model(X_batch)
                total_loss += criterion(pred_u, pred_l,
                                        u_batch, l_batch).item()
                n += 1
        return total_loss / max(n, 1)
