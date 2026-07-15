"""
深度学习波动区间预测 (Phase 4A - DL Range V2)

预测未来5天的价格波动区间:
    upper = max(High_t+1, ..., High_t+5) / Close_t - 1
    lower = min(Low_t+1, ..., Low_t+5) / Close_t - 1

V2 改进:
    1. 归一化目标: 除以 rv*sqrt(5), 模型预测 sigma multiplier
    2. Conformal calibration: 独立校准集 (非训练val集)
    3. 多种子集成: 提高稳定性

V3 改进 (借鉴 Qlib 模型架构):
    4. 支持多模型: LSTM / Transformer / ALSTM
    5. Transformer: positional encoding + self-attention
    6. ALSTM: FC投影 → RNN → 双路注意力 (Qlib 排名前列)

用法:
    from src.models.dl_range_predictor import DLRangePredictor
    # LSTM (默认, 向后兼容)
    predictor = DLRangePredictor(model_type="lstm")
    # Transformer
    predictor = DLRangePredictor(model_type="transformer")
    # ALSTM
    predictor = DLRangePredictor(model_type="alstm")
"""

import math
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

from src.models.dl_fair_value import SELECTED_FEATURES

logger = logging.getLogger(__name__)


def select_features(df: pd.DataFrame) -> list:
    return [f for f in SELECTED_FEATURES if f in df.columns]


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


class RangeLSTM(nn.Module):
    """LSTM 预测波动区间 sigma multiplier。"""

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
# RangeTransformer (改编自 Qlib Transformer)
# ======================================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码 (来自 Qlib / Attention Is All You Need)。"""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, 1, d_model]
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0)]


class RangeTransformer(nn.Module):
    """
    Transformer 预测波动区间 (改编自 Qlib Transformer)。

    输入: [batch, seq_len, n_features]
    输出: (upper, lower)

    与 Qlib 原版的区别:
    - 输入已是 [batch, seq_len, feat] 不需要 reshape
    - 双头输出 (upper/lower) 替代单 score 输出
    - 加了 input BatchNorm
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.feature_layer = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=False,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.upper_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.lower_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, feat]
        batch, seq_len, feat = x.shape

        # BatchNorm (per-feature)
        x_flat = x.reshape(-1, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        # 线性投影到 d_model
        x = self.feature_layer(x)  # [batch, seq_len, d_model]

        # Transformer 需要 [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [seq_len, batch, d_model]

        # 取最后一个时间步
        last = x[-1]  # [batch, d_model]

        upper = self.upper_head(last).squeeze(-1)
        lower = self.lower_head(last).squeeze(-1)
        return upper, lower


# ======================================================================
# RangeALSTM (改编自 Qlib ALSTM)
# ======================================================================

class RangeALSTM(nn.Module):
    """
    Attention-LSTM 预测波动区间 (改编自 Qlib ALSTM)。

    与 Qlib 原版的区别:
    - 输入已是 [batch, seq_len, feat] 不需要 reshape
    - 双头输出 (upper/lower) 替代单 score 输出
    - 加了 input BatchNorm
    - FC投影 → LSTM → 双路注意力 (attention context ⊕ last hidden)
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 rnn_type: str = "LSTM"):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_bn = nn.BatchNorm1d(n_features)

        # FC 投影层 (Qlib 特色: 先做线性变换再进 RNN)
        self.fc_in = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Tanh(),
        )

        # RNN
        rnn_cls = getattr(nn, rnn_type.upper())
        self.rnn = rnn_cls(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # 注意力网络
        self.att_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # 双头输出 (attention context + last hidden 拼接, 所以是 hidden*2)
        self.upper_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.lower_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, feat]
        batch, seq_len, feat = x.shape

        # BatchNorm
        x_flat = x.reshape(-1, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        # FC 投影
        x = self.fc_in(x)  # [batch, seq_len, hidden]

        # RNN
        rnn_out, _ = self.rnn(x)  # [batch, seq_len, hidden]

        # 双路注意力 (Qlib ALSTM 特色)
        att_scores = self.att_net(rnn_out)  # [batch, seq_len, 1]
        att_context = torch.sum(rnn_out * att_scores, dim=1)  # [batch, hidden]
        last_hidden = rnn_out[:, -1, :]  # [batch, hidden]

        # 拼接: attention 加权和 + 最后隐状态
        combined = torch.cat([att_context, last_hidden], dim=1)  # [batch, hidden*2]

        upper = self.upper_head(combined).squeeze(-1)
        lower = self.lower_head(combined).squeeze(-1)
        return upper, lower


# ======================================================================
# 模型工厂
# ======================================================================

MODEL_REGISTRY = {
    "lstm": RangeLSTM,
    "transformer": RangeTransformer,
    "alstm": RangeALSTM,
}


def create_range_model(model_type: str, n_features: int,
                       hidden_size: int = 64, num_layers: int = 2,
                       dropout: float = 0.2, **kwargs) -> nn.Module:
    """根据 model_type 创建对应的区间预测模型。"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]

    if model_type == "transformer":
        nhead = kwargs.get("nhead", 4)
        # d_model 必须能被 nhead 整除
        d_model = hidden_size
        if d_model % nhead != 0:
            d_model = (d_model // nhead + 1) * nhead
        return cls(n_features, d_model=d_model, nhead=nhead,
                   num_layers=num_layers, dropout=dropout)
    elif model_type == "alstm":
        rnn_type = kwargs.get("rnn_type", "LSTM")
        return cls(n_features, hidden_size=hidden_size,
                   num_layers=num_layers, dropout=dropout,
                   rnn_type=rnn_type)
    else:
        return cls(n_features, hidden_size=hidden_size,
                   num_layers=num_layers, dropout=dropout)


class DLRangePredictor:
    """
    深度学习波动区间预测器 (V3: 多模型 + 归一化 + 独立校准 + 集成)。

    训练流程:
        1. 目标归一化: upper/lower 除以 rv_scale
        2. 在 train 上训练, val 上 early stopping
        3. 在独立 cal 集上做 conformal calibration
        4. n_ensemble 个模型取平均

    支持模型: lstm, transformer, alstm, ensemble (默认, LSTM+Transformer)
    """

    def __init__(self, seq_len: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 150, batch_size: int = 64,
                 patience: int = 20, device: str = "auto",
                 q_upper: float = 0.85, q_lower: float = 0.15,
                 n_ensemble: int = 3,
                 cal_target_cov: float = 0.80,
                 model_type: str = "ensemble",
                 **model_kwargs):
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
        self.model_type = model_type
        self.model_kwargs = model_kwargs

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
        """
        训练集成模型。

        cal_* 参数: 独立校准集 (不用于训练/early stopping)
        """
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

        # 集成训练
        # ensemble 模式: 每种架构各训练 n_ensemble 个
        # 单模型模式: 同架构训练 n_ensemble 个
        if self.model_type == "ensemble":
            arch_list = ["lstm", "transformer"]
        else:
            arch_list = [self.model_type]

        self.models = []
        for arch in arch_list:
            for seed_i in range(self.n_ensemble):
                torch.manual_seed(42 + seed_i * 7)
                np.random.seed(42 + seed_i * 7)

                model = create_range_model(
                    arch, self.n_features,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    **self.model_kwargs,
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

        # Conformal calibration on SEPARATE cal set
        if cal_features is not None:
            self._calibrate(cal_features, cal_upper, cal_lower, cal_rv_scale)

        return self

    def _calibrate(self, cal_features, cal_upper, cal_lower, cal_rv_scale):
        """
        Conformal calibration on independent calibration set.

        找到最小的 margin 使得校准集上覆盖率 >= cal_target_cov。
        """
        pred_u, pred_l = self._predict_raw(cal_features, cal_rv_scale)
        n_pred = len(pred_u)

        actual_u = cal_upper[self.seq_len - 1:][:n_pred]
        actual_l = cal_lower[self.seq_len - 1:][:n_pred]

        # 过滤 NaN (cal集尾部可能缺少未来数据)
        valid = ~(np.isnan(actual_u) | np.isnan(actual_l))
        actual_u, actual_l = actual_u[valid], actual_l[valid]
        pred_u, pred_l = pred_u[valid], pred_l[valid]

        # 上限残差: actual - pred (正 = 突破)
        upper_residual = actual_u - pred_u
        # 下限残差: pred - actual (正 = 突破)
        lower_residual = pred_l - actual_l

        # 找最小 margin 使得各自覆盖率 >= sqrt(target_cov)
        # (因为 total_cov ≈ upper_cov × lower_cov)
        per_side_target = np.sqrt(self.cal_target_cov)
        q_pct = per_side_target * 100

        self.cal_upper_margin = float(np.percentile(upper_residual, q_pct))
        self.cal_lower_margin = float(np.percentile(lower_residual, q_pct))

        # 允许负 margin (收紧范围), 但不要收太多
        self.cal_upper_margin = max(self.cal_upper_margin, -0.5)
        self.cal_lower_margin = max(self.cal_lower_margin, -0.5)

    def _predict_raw(self, features, rv_scale):
        """集成预测 (归一化空间 → 乘回 rv_scale), 不加 margin。"""
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

        # 集成平均
        avg_upper_norm = np.mean(all_uppers, axis=0)
        avg_lower_norm = np.mean(all_lowers, axis=0)

        # 乘回 rv_scale
        rv_aligned = rv_scale[self.seq_len - 1:]
        rv_safe = np.clip(rv_aligned, 0.5, None)

        return avg_upper_norm * rv_safe, avg_lower_norm * rv_safe

    def predict(self, features, rv_scale):
        """预测并应用 conformal margin。"""
        pred_u, pred_l = self._predict_raw(features, rv_scale)
        pred_u = pred_u + self.cal_upper_margin
        pred_l = pred_l - self.cal_lower_margin
        return pred_u, pred_l

    def save(self, path):
        """保存模型权重 + scaler + margins + 架构信息 (支持多架构 ensemble)."""
        import pickle
        if self.model_type == "ensemble":
            arch_list = ["lstm", "transformer"]
        else:
            arch_list = [self.model_type]
        model_types = [a for a in arch_list for _ in range(self.n_ensemble)]

        state = {
            "version": 2,
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "q_upper": self.q_upper,
            "q_lower": self.q_lower,
            "n_ensemble": self.n_ensemble,
            "cal_target_cov": self.cal_target_cov,
            "model_type": self.model_type,
            "model_types": model_types,
            "model_kwargs": self.model_kwargs,
            "cal_upper_margin": self.cal_upper_margin,
            "cal_lower_margin": self.cal_lower_margin,
            "scaler": self.scaler,
            "model_states": [m.state_dict() for m in self.models],
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        """加载已训练的集成预测器 (支持 LSTM/Transformer/ALSTM 混合)."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        pred = cls(
            seq_len=state["seq_len"],
            hidden_size=state["hidden_size"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            q_upper=state.get("q_upper", 0.85),
            q_lower=state.get("q_lower", 0.15),
            n_ensemble=state.get("n_ensemble", 3),
            cal_target_cov=state.get("cal_target_cov", 0.80),
            model_type=state.get("model_type", "ensemble"),
            **state.get("model_kwargs", {}),
        )
        pred.n_features = state["n_features"]
        pred.scaler = state["scaler"]
        pred.cal_upper_margin = state["cal_upper_margin"]
        pred.cal_lower_margin = state["cal_lower_margin"]

        model_types = state.get("model_types")
        if model_types is None:
            model_types = ["lstm"] * len(state["model_states"])

        for mt, ms in zip(model_types, state["model_states"]):
            model = create_range_model(
                mt, state["n_features"],
                hidden_size=state["hidden_size"],
                num_layers=state["num_layers"],
                dropout=state["dropout"],
                **state.get("model_kwargs", {}),
            )
            model.load_state_dict(ms)
            model.to(pred.device)
            pred.models.append(model)

        return pred

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
