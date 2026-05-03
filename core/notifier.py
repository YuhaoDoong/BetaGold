"""通知模块 (NotificationDispatcher) — 信号变更时推送通知.

设计目标 (v3.7.19 骨架):
  当前: 用户手动打开 dashboard 查看信号. 用于实盘验证胜率.
  未来: 服务器持续运行模型, 信号变更时自动推送 (telegram / email / sms / webhook).

接入方式 (后续):
  from core.notifier import Notifier
  notifier = Notifier(channels=["telegram"])
  notifier.notify_signal_change(
      old_signal=None,
      new_signal="STRADDLE",
      details={"score": 7, "rv": 17.8, "fomc_in_days": 1}
  )

支持渠道 (按需添加):
  - telegram: bot_token + chat_id
  - email: smtp + 收件人
  - sms: twilio
  - webhook: 自定义 URL POST
  - file: 本地 log (默认, 已实现)

模块化原则:
  - Channel 是抽象基类, 子类实现 send()
  - Notifier 统一接口 dispatch 到所有启用的 channel
  - 信号变更检测在 NotifierState 中, 跨重启用 parquet 持久化
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path


# ── 数据类 ──
@dataclass
class SignalSnapshot:
    """单次信号快照, 用于跨重启对比."""
    timestamp: str         # ISO datetime (SGT)
    asset: str             # 'GLD' / 'SLV'
    chosen: str            # 'BUY CALL' / 'STRADDLE' / 'SHORT_VOL' / etc.
    score: int             # straddle/short_vol score
    rv: float              # 当时 RV
    rv_pctile: float       # 当时 RV %tile
    bp_low: float          # 当时 bp_low
    is_us_session: bool    # 是否 US 期权时段中
    extras: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "chosen": self.chosen,
            "score": self.score,
            "rv": self.rv,
            "rv_pctile": self.rv_pctile,
            "bp_low": self.bp_low,
            "is_us_session": self.is_us_session,
            "extras": self.extras or {},
        }


# ── Channel 抽象基类 ──
class Channel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def send(self, subject: str, body: str, **meta) -> bool:
        """发送通知. 返回 True 成功, False 失败."""
        ...


class FileChannel(Channel):
    """默认实现: 写入本地 log 文件 (~/notifier.log)."""
    name = "file"

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or str(Path.home() / "GoldDash_notifier.log")

    def send(self, subject: str, body: str, **meta) -> bool:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                ts = datetime.now().isoformat(timespec="seconds")
                f.write(f"[{ts}] {subject}\n{body}\n---\n")
            return True
        except Exception:
            return False


class TelegramChannel(Channel):
    """Telegram 推送 (待实现, 仅占位).

    使用方式 (后续):
        from telegram import Bot
        self.bot = Bot(token=bot_token)
        self.bot.send_message(chat_id=chat_id, text=body)
    """
    name = "telegram"

    def __init__(self, bot_token: Optional[str] = None,
                 chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get("TG_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TG_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)

    def send(self, subject: str, body: str, **meta) -> bool:
        if not self._enabled:
            return False
        try:
            import urllib.parse, urllib.request
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": f"*{subject}*\n\n{body}",
                "parse_mode": "Markdown",
            }).encode()
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[telegram] {e}")
            return False


class EmailChannel(Channel):
    """SMTP 邮件推送 (v3.7.58 实装)."""
    name = "email"

    def __init__(self, smtp_host: Optional[str] = None,
                 smtp_port: int = 587,
                 smtp_user: Optional[str] = None,
                 smtp_pass: Optional[str] = None,
                 from_addr: Optional[str] = None,
                 to_addrs: Optional[List[str]] = None):
        self.smtp_host = smtp_host or os.environ.get("SMTP_HOST", "")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.environ.get("SMTP_USER", "")
        self.smtp_pass = smtp_pass or os.environ.get("SMTP_PASS", "")
        self.from_addr = from_addr or os.environ.get("SMTP_FROM",
                                                       self.smtp_user)
        env_to = os.environ.get("SMTP_TO", "")
        self.to_addrs = to_addrs or ([a.strip() for a in env_to.split(",")
                                        if a.strip()])
        self._enabled = bool(self.smtp_host and self.to_addrs and self.smtp_user)

    def send(self, subject: str, body: str, **meta) -> bool:
        if not self._enabled:
            return False
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as s:
                s.starttls()
                if self.smtp_pass:
                    s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
            return True
        except Exception as e:
            print(f"[email] {e}")
            return False


class WebhookChannel(Channel):
    """通用 webhook (POST JSON, v3.7.58 实装)."""
    name = "webhook"

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("WEBHOOK_URL", "")
        self._enabled = bool(self.url)

    def send(self, subject: str, body: str, **meta) -> bool:
        if not self._enabled:
            return False
        try:
            import urllib.request
            payload = json.dumps({
                "subject": subject,
                "body": body,
                **{k: v for k, v in meta.items()
                   if isinstance(v, (str, int, float, bool, list, dict))},
            }).encode()
            req = urllib.request.Request(
                self.url, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 201, 204)
        except Exception as e:
            print(f"[webhook] {e}")
            return False


# ── Notifier 主类 ──
class Notifier:
    """统一通知调度. dispatch 到所有启用的 channels.

    Args:
        channels: List[Channel] 实例; 默认 [FileChannel()]
        state_path: 信号状态持久化路径 (parquet/json)

    使用:
        n = Notifier()
        n.notify_signal_change(prev_snap, new_snap)
    """
    def __init__(self,
                 channels: Optional[List[Channel]] = None,
                 state_path: Optional[str] = None):
        self.channels = channels or [FileChannel()]
        self.state_path = state_path or str(
            Path.home() / "GoldDash_notifier_state.json")
        self._last_snap: Optional[SignalSnapshot] = self._load_state()

    def _load_state(self) -> Optional[SignalSnapshot]:
        try:
            with open(self.state_path) as f:
                d = json.load(f)
            return SignalSnapshot(**d)
        except Exception:
            return None

    def _save_state(self, snap: SignalSnapshot):
        try:
            with open(self.state_path, "w") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception:
            pass

    def is_signal_changed(self, new_snap: SignalSnapshot) -> bool:
        """检测信号是否实质变化 (chosen 不同 OR score 上升)."""
        if self._last_snap is None:
            return new_snap.chosen != "—" and new_snap.chosen != ""
        if self._last_snap.chosen != new_snap.chosen:
            return True
        # 同 chosen 但 score 显著上升 (≥2 分) 也提醒
        if new_snap.score - self._last_snap.score >= 2:
            return True
        return False

    def notify_signal_change(self, new_snap: SignalSnapshot,
                              force: bool = False) -> Dict[str, bool]:
        """检测变化并 dispatch 到所有 channels.

        Args:
            new_snap: 当前最新信号快照
            force: True 时跳过变更检测, 强制发送

        Returns:
            {channel_name: success_bool}
        """
        if not force and not self.is_signal_changed(new_snap):
            return {}

        prev_chosen = self._last_snap.chosen if self._last_snap else "—"
        subject = f"[GoldDash] {new_snap.asset} 信号: {prev_chosen} → {new_snap.chosen}"
        body = (
            f"时间: {new_snap.timestamp}\n"
            f"资产: {new_snap.asset}\n"
            f"新信号: {new_snap.chosen}\n"
            f"评分: {new_snap.score}\n"
            f"RV: {new_snap.rv:.1f}%, RV %tile: {new_snap.rv_pctile:.0%}\n"
            f"bp_low: {new_snap.bp_low:.3f}\n"
            f"US 时段: {'是' if new_snap.is_us_session else '否'}\n"
        )
        if new_snap.extras:
            body += f"额外: {json.dumps(new_snap.extras, ensure_ascii=False)}\n"

        results = {}
        for ch in self.channels:
            results[ch.name] = ch.send(subject, body,
                                         snapshot=new_snap.to_dict())

        # 持久化最新快照 (即使发送失败也更新, 避免重复通知)
        self._last_snap = new_snap
        self._save_state(new_snap)
        return results


# ── Demo / 测试入口 ──
if __name__ == "__main__":
    n = Notifier()
    snap = SignalSnapshot(
        timestamp=datetime.now().isoformat(),
        asset="GLD",
        chosen="STRADDLE",
        score=7,
        rv=17.8, rv_pctile=0.49,
        bp_low=0.216,
        is_us_session=False,
        extras={"fomc_in_days": 1, "next_event": "FOMC 03/18"},
    )
    res = n.notify_signal_change(snap, force=True)
    print(f"Sent: {res}")
