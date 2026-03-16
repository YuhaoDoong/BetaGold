"""1h 多粒度区间预测模型.

架构同 v1.0 的 RangeLSTM (BiLSTM + Attention + 双头),
但输入为 1h 多粒度特征, 同时预测多个时间尺度.

变化:
  - seq_len: 48 (约2个交易日的1h K线)
  - 多目标: 7h (日内) + 35h (5天) 同时预测
  - RV归一化: 用 rv_10h 替代 rv_10d
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


class RangeLSTM1h(nn.Module):
    """BiLSTM + Attention, 多时间尺度双头输出."""

    def __init__(self, n_features, hidden_size=64, num_layers=2,
                 dropout=0.2, n_horizons=2):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.Linear(hidden_size, 1)

        # 每个 horizon 有独立的 upper/lower head
        self.upper_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Softplus(),
            ) for _ in range(n_horizons)
        ])
        self.lower_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            ) for _ in range(n_horizons)
        ])

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        b, s, f = x.shape
        x_bn = self.bn(x.reshape(-1, f)).reshape(b, s, f)
        out, _ = self.lstm(x_bn)  # (batch, seq_len, hidden)

        # Attention
        weights = torch.softmax(self.attn(out), dim=1)  # (batch, seq_len, 1)
        context = (out * weights).sum(dim=1)  # (batch, hidden)

        uppers = [head(context) for head in self.upper_heads]
        lowers = [head(context) for head in self.lower_heads]

        return uppers, lowers  # lists of (batch, 1)


class RangeDataset1h(Dataset):
    """Sequence dataset for 1h range prediction."""

    def __init__(self, features, targets, rv_scale, seq_len=48):
        """
        features: np.array (n_samples, n_features)
        targets: list of (upper_pct, lower_pct) arrays, one per horizon
        rv_scale: np.array (n_samples,) for normalization
        """
        self.features = features.astype(np.float32)
        self.targets = targets  # list of (upper, lower) tuples
        self.rv_scale = np.clip(rv_scale, 0.5, None).astype(np.float32)
        self.seq_len = seq_len

        # Valid indices: need seq_len lookback + valid target
        self.valid = []
        for i in range(seq_len, len(features)):
            if all(not np.isnan(t[0][i]) and not np.isnan(t[1][i])
                   for t in targets):
                self.valid.append(i)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        x = self.features[i - self.seq_len: i]
        rv = self.rv_scale[i]

        # Normalize targets by rv
        upper_targets = [t[0][i] / rv for t in self.targets]
        lower_targets = [t[1][i] / rv for t in self.targets]

        return (
            torch.from_numpy(x),
            [torch.tensor(u, dtype=torch.float32) for u in upper_targets],
            [torch.tensor(l, dtype=torch.float32) for l in lower_targets],
        )


def quantile_loss(pred, target, q):
    """Pinball loss."""
    e = target - pred
    return torch.mean(torch.max(q * e, (q - 1) * e))


class DLRangePredictor1h:
    """训练+预测封装. 多时间尺度, 3种子集成, Conformal校准."""

    def __init__(self, seq_len=48, hidden_size=64, num_layers=2,
                 dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=150, batch_size=64, patience=20,
                 q_upper=0.85, q_lower=0.15,
                 n_ensemble=3, cal_target_cov=0.80,
                 horizons=(7, 35)):
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
        self.horizons = horizons
        self.n_horizons = len(horizons)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.models = []
        self.scaler = None
        self.cal_margins = None  # per-horizon

    def _train_one(self, train_ds, val_ds, seed):
        """Train one model with given seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_feat = train_ds.features.shape[1]
        model = RangeLSTM1h(
            n_feat, self.hidden_size, self.num_layers,
            self.dropout, self.n_horizons,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for x, u_targets, l_targets in train_loader:
                x = x.to(self.device)
                uppers, lowers = model(x)

                loss = 0
                for h_idx in range(self.n_horizons):
                    u_t = u_targets[h_idx].to(self.device)
                    l_t = l_targets[h_idx].to(self.device)
                    loss += quantile_loss(uppers[h_idx].squeeze(), u_t,
                                          self.q_upper)
                    loss += quantile_loss(lowers[h_idx].squeeze(), l_t,
                                          self.q_lower)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, u_targets, l_targets in val_loader:
                    x = x.to(self.device)
                    uppers, lowers = model(x)
                    loss = 0
                    for h_idx in range(self.n_horizons):
                        u_t = u_targets[h_idx].to(self.device)
                        l_t = l_targets[h_idx].to(self.device)
                        loss += quantile_loss(
                            uppers[h_idx].squeeze(), u_t, self.q_upper)
                        loss += quantile_loss(
                            lowers[h_idx].squeeze(), l_t, self.q_lower)
                    val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        model.load_state_dict(best_state)
        return model

    def _predict(self, model, features_scaled, rv_scale):
        """Predict with one model."""
        model.eval()
        n = len(features_scaled)
        results = {h: {"upper": [], "lower": []} for h in range(self.n_horizons)}

        with torch.no_grad():
            for i in range(self.seq_len, n):
                x = torch.from_numpy(
                    features_scaled[i - self.seq_len: i].astype(np.float32)
                ).unsqueeze(0).to(self.device)

                uppers, lowers = model(x)
                rv = max(rv_scale[i], 0.5)

                for h_idx in range(self.n_horizons):
                    results[h_idx]["upper"].append(
                        uppers[h_idx].item() * rv)
                    results[h_idx]["lower"].append(
                        lowers[h_idx].item() * rv)

        return results

    def fit_predict(self, features, targets_list, rv_scale, dates,
                    train_end_idx, val_size=350, cal_size=175):
        """Walk-forward fit + predict.

        Args:
            features: np.array (n, n_feat)
            targets_list: list of (upper_pct_arr, lower_pct_arr) per horizon
            rv_scale: np.array (n,)
            dates: DatetimeIndex
            train_end_idx: int, 训练数据截止 index
            val_size: validation size (in 1h bars)
            cal_size: calibration size

        Returns: dict of {horizon_idx: DataFrame with pred_upper/lower_pct}
        """
        # Scale features
        self.scaler = RobustScaler()
        feat_train = features[:train_end_idx]
        self.scaler.fit(feat_train)
        feat_scaled = self.scaler.transform(features)

        # Replace NaN
        feat_scaled = np.nan_to_num(feat_scaled, 0)

        # Split
        cal_start = train_end_idx - cal_size
        val_start = cal_start - val_size
        assert val_start > self.seq_len, \
            f"Not enough training data: val_start={val_start}, seq_len={self.seq_len}"

        train_ds = RangeDataset1h(
            feat_scaled[:val_start], targets_list, rv_scale, self.seq_len)
        val_ds = RangeDataset1h(
            feat_scaled[:cal_start], targets_list, rv_scale, self.seq_len)
        # val_ds covers val_start..cal_start because valid indices > seq_len

        print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

        # Train ensemble
        self.models = []
        for i in range(self.n_ensemble):
            seed = 42 + i * 7
            print(f"  Training model {i+1}/{self.n_ensemble} (seed={seed})...")
            model = self._train_one(train_ds, val_ds, seed)
            self.models.append(model)

        # Predict on calibration set + test set
        all_preds = []
        for model in self.models:
            preds = self._predict(model, feat_scaled, rv_scale)
            all_preds.append(preds)

        # Ensemble average
        ens_preds = {}
        for h_idx in range(self.n_horizons):
            upper_avg = np.mean(
                [p[h_idx]["upper"] for p in all_preds], axis=0)
            lower_avg = np.mean(
                [p[h_idx]["lower"] for p in all_preds], axis=0)
            ens_preds[h_idx] = {
                "upper": upper_avg,
                "lower": lower_avg,
            }

        # Conformal calibration on cal set
        self.cal_margins = {}
        cal_range = range(cal_start - self.seq_len,
                          train_end_idx - self.seq_len)

        for h_idx in range(self.n_horizons):
            actual_u = np.array([targets_list[h_idx][0][i + self.seq_len]
                                 for i in cal_range])
            actual_l = np.array([targets_list[h_idx][1][i + self.seq_len]
                                 for i in cal_range])
            pred_u = ens_preds[h_idx]["upper"][
                cal_start - self.seq_len: train_end_idx - self.seq_len]
            pred_l = ens_preds[h_idx]["lower"][
                cal_start - self.seq_len: train_end_idx - self.seq_len]

            valid = ~np.isnan(actual_u) & ~np.isnan(actual_l)
            if valid.sum() > 10:
                u_resid = actual_u[valid] - np.array(pred_u)[valid]
                l_resid = np.array(pred_l)[valid] - actual_l[valid]
                q = np.sqrt(self.cal_target_cov) * 100
                u_margin = max(np.percentile(u_resid, q), -0.5)
                l_margin = max(np.percentile(l_resid, q), -0.5)
            else:
                u_margin = l_margin = 0

            self.cal_margins[h_idx] = (u_margin, l_margin)
            print(f"  Horizon {self.horizons[h_idx]}h: "
                  f"cal margins = (+{u_margin:.3f}, +{l_margin:.3f})")

        # Apply calibration to test predictions
        test_start = train_end_idx - self.seq_len
        results = {}
        for h_idx in range(self.n_horizons):
            u_m, l_m = self.cal_margins[h_idx]
            pred_u = ens_preds[h_idx]["upper"][test_start:]
            pred_l = ens_preds[h_idx]["lower"][test_start:]

            # Create output DataFrame
            out_dates = dates[train_end_idx:]
            n_out = min(len(pred_u), len(out_dates))
            nh = self.horizons[h_idx]
            results[h_idx] = pd.DataFrame({
                f"pred_{nh}h_upper_pct": np.array(pred_u[:n_out]) + u_m,
                f"pred_{nh}h_lower_pct": np.array(pred_l[:n_out]) - l_m,
            }, index=out_dates[:n_out])

        return results
