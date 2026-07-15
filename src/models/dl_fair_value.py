"""
深度学习公允价格预测 (Phase 4A - DL Enhancement)

用深度学习预测 "未来5天价格相对于当前价格的方向和幅度":
    target = log(price_t+5 / price_t)

输入: 过去 seq_len 天的多因子时序
输出: 预测的 5d log return → fair_value = current_price × exp(predicted)

架构:
- GoldLSTM: LSTM + Attention + Dense
- GoldTransformer: Transformer Encoder + Mean Pool + Dense

用法:
    from src.models.dl_fair_value import DLFairValuePredictor
    predictor = DLFairValuePredictor(model_type="transformer")
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

# 精选特征: 避免高度冗余, 保留信息量最大的
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
    # ---- Alpha158 因子 (经 IC/ICIR 筛选, |ICIR| > 0.7) ----
    # 价格位置 (ICIR 1.0~1.4, 最强新因子组)
    "qtld_60d", "qtlu_60d", "qtld_30d", "qtlu_30d",
    # 涨跌累积 (ICIR ~1.2, 类 RSI 但多窗口)
    "sumd_60d", "sump_60d", "sumd_30d",
    # 涨跌天数统计 (ICIR ~1.0)
    "cntd_60d", "cntn_60d",
    # 价格区间位置 (ICIR ~0.9)
    "rsv_60d", "rsv_30d",
    # 趋势斜率 (ICIR ~0.9)
    "beta_30d", "beta_60d", "beta_20d",
    # 高低点时序 - Aroon 类 (ICIR ~0.78)
    "imxd_60d", "imxd_30d",
    # K线形态 (ICIR ~0.7)
    "kbar_kmid", "kbar_klen",
]


def select_features(df: pd.DataFrame) -> list:
    """从 DataFrame 中选择可用的特征列。"""
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    return available


# ======================================================================
# Dataset
# ======================================================================

class GoldSequenceDataset(Dataset):
    """时序数据集: (seq_len, n_features) → target"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = 20):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.targets) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(y)


# ======================================================================
# Model
# ======================================================================

class Attention(nn.Module):
    """简单的 temporal attention: 对 LSTM 各时步加权。"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)     # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1),
                            lstm_out).squeeze(1)   # (batch, hidden)
        return context, weights


class GoldLSTM(nn.Module):
    """
    LSTM + Attention 预测 5d log return。

    Input:  (batch, seq_len, n_features)
    Output: (batch, 1) — predicted 5d log return
    """

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
        self.attention = Attention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, feat = x.shape
        # BatchNorm on feature dimension
        x_flat = x.reshape(-1, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        context, _ = self.attention(lstm_out)  # (batch, hidden)
        out = self.head(context).squeeze(-1)   # (batch,)
        return out


class GoldTransformer(nn.Module):
    """
    Transformer Encoder 预测 5d log return。

    Input:  (batch, seq_len, n_features)
    Output: (batch,) — predicted 5d log return

    架构:
    - Input projection: n_features → d_model
    - Learnable positional embedding
    - TransformerEncoder (multi-head self-attention)
    - Mean pooling over time → dense head
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.2,
                 max_seq_len: int = 60):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(n_features)

        # Project input features to d_model dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, feat = x.shape

        # BatchNorm
        x_flat = x.reshape(-1, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional embedding
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)

        # Transformer encode
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = self.layer_norm(x)

        # Mean pool over time
        x = x.mean(dim=1)  # (batch, d_model)

        # Output
        out = self.head(x).squeeze(-1)  # (batch,)
        return out


def build_model(model_type: str, n_features: int, **kwargs) -> nn.Module:
    """根据 model_type 创建模型。"""
    if model_type == "lstm":
        return GoldLSTM(
            n_features,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_type == "transformer":
        return GoldTransformer(
            n_features,
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 3),
            dim_feedforward=kwargs.get("dim_feedforward", 128),
            dropout=kwargs.get("dropout", 0.2),
            max_seq_len=kwargs.get("max_seq_len", 60),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ======================================================================
# Predictor (high-level API)
# ======================================================================

class DLFairValuePredictor:
    """
    深度学习公允价格预测器。

    封装: 数据预处理 → 模型训练 → 预测 → 生成公允价格

    用法:
        predictor = DLFairValuePredictor(seq_len=20, hidden_size=64)
        predictor.fit(train_features, train_targets)
        predictions = predictor.predict(test_features)
        fair_values = current_prices * np.exp(predictions)
    """

    def __init__(self, seq_len: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 50, batch_size: int = 64,
                 patience: int = 10, device: str = "auto",
                 model_type: str = "lstm",
                 # Transformer-specific
                 d_model: int = 64, nhead: int = 4,
                 dim_feedforward: int = 128):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model_type = model_type
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

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
        self.model = None
        self.n_features = None
        self.train_losses = []

    def fit(self, features: np.ndarray, targets: np.ndarray,
            val_features: np.ndarray = None, val_targets: np.ndarray = None,
            verbose: bool = False):
        """
        训练模型。

        features: (n_samples, n_features)
        targets: (n_samples,) — 5d log returns
        """
        self.n_features = features.shape[1]

        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        X_scaled = np.nan_to_num(X_scaled, 0)

        # Create dataset
        dataset = GoldSequenceDataset(X_scaled, targets, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True)

        # Validation
        val_loader = None
        if val_features is not None and val_targets is not None:
            X_val_scaled = self.scaler.transform(val_features)
            X_val_scaled = np.nan_to_num(X_val_scaled, 0)
            val_dataset = GoldSequenceDataset(
                X_val_scaled, val_targets, self.seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False)

        # Model
        self.model = build_model(
            self.model_type, self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            max_seq_len=self.seq_len + 10,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.train_losses = []

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train)

            # Validation
            val_loss = avg_train
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion)

            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: "
                      f"train={avg_train:.6f} val={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch+1}")
                    break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测 5d log returns。

        features: (n_samples, n_features)
        返回: (n_samples - seq_len + 1,) 的预测值
        """
        self.model.eval()
        X_scaled = self.scaler.transform(features)
        X_scaled = np.nan_to_num(X_scaled, 0)

        dataset = GoldSequenceDataset(
            X_scaled, np.zeros(len(X_scaled)), self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False)

        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)

    def _eval_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(X_batch)
                total_loss += criterion(pred, y_batch).item()
                n += 1
        return total_loss / max(n, 1)
