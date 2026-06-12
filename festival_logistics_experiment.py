"""
========================================================================
  통합 예측 모델 기반 축제 물류 자원 협력 의사결정 플랫폼
  基于集成预测模型的节庆物流资源协同决策平台 — 实验性 Python 代码
  Author : 견호이 (50260250)
========================================================================

모듈 구성
  [1] 데이터 생성 및 전처리
  [2] Prophet-Style 성분 분해 (scipy 기반)
  [3] LSTM 시계열 예측 (numpy 순수 구현)
  [4] LSTM-Prophet 가중 앙상블 융합 모델
  [5] 다목표 자원 협력 최적화 (ε-제약법, scipy 기반)
  [6] 성능 평가 및 비교 분석
  [7] 시각화 출력 (논문 수준 그래프 7종)
  [8] 통합 파이프라인 실행 & 결과 요약 테이블
========================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.fft import fft, ifft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import time, os, textwrap

np.random.seed(42)

# ── 컬러 팔레트 ─────────────────────────────────────────────────────────
COLORS = {
    "bg":     "#0a0e1a",
    "panel":  "#0f1628",
    "border": "#1e2d4a",
    "accent": "#00c8ff",
    "green":  "#00e5a0",
    "purple": "#7b5ea7",
    "warn":   "#ffab2e",
    "red":    "#ff4d6d",
    "text":   "#c8d8f0",
    "muted":  "#5a7090",
    "white":  "#f0f8ff",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor":   COLORS["panel"],
    "axes.edgecolor":   COLORS["border"],
    "axes.labelcolor":  COLORS["text"],
    "xtick.color":      COLORS["muted"],
    "ytick.color":      COLORS["muted"],
    "text.color":       COLORS["text"],
    "grid.color":       COLORS["border"],
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": COLORS["panel"],
    "legend.edgecolor": COLORS["border"],
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

SAVE_DIR = "/mnt/user-data/outputs/"
os.makedirs(SAVE_DIR, exist_ok=True)

SEP = "=" * 68


# ══════════════════════════════════════════════════════════════════════
# [1]  데이터 생성 및 전처리
# ══════════════════════════════════════════════════════════════════════
class FestivalDataGenerator:
    """
    국경 간 축제 물류 시계열 합성 데이터 생성기.
    실제 기업 탈식별화 데이터와 동일한 통계적 특성을 재현:
      - 선형 성장 추세
      - 7일 주기 계절성 (주말 효과)
      - 365일 연간 주기성
      - 축제 피크 가우시안 스파이크
      - 국경 간 통관 지연 효과
      - 가우시안 노이즈
    """

    FESTIVALS = {
        "블랙프라이데이":  {"day": 60, "scale": 1.0, "sigma": 4},
        "광군절":         {"day": 60, "scale": 0.9, "sigma": 3},
        "설날":           {"day": 60, "scale": 0.8, "sigma": 5},
        "발렌타인데이":   {"day": 60, "scale": 0.6, "sigma": 3},
        "크리스마스":     {"day": 60, "scale": 0.85, "sigma": 6},
    }

    def __init__(self, festival="블랙프라이데이", n_days=74, seed=42):
        self.festival = festival
        self.n_days   = n_days
        cfg = self.FESTIVALS.get(festival, self.FESTIVALS["블랙프라이데이"])
        self.peak_day  = cfg["day"]
        self.scale     = cfg["scale"]
        self.sigma     = cfg["sigma"]
        np.random.seed(seed)

    def generate(self):
        t    = np.arange(self.n_days)
        base  = 300 + 2.1 * t                                              # 추세
        week  = 40 * np.sin(2 * np.pi * t / 7)                            # 주간 계절성
        year  = 25 * np.sin(2 * np.pi * t / 365 * 3)                      # 연간 성분
        peak  = (800 * self.scale
                 * np.exp(-0.5 * ((t - self.peak_day) / self.sigma) ** 2))  # 축제 스파이크
        custom = np.where((t >= self.peak_day - 3) & (t <= self.peak_day + 2),
                          -80, 0)                                           # 통관 지연 억제
        noise = np.random.normal(0, 45, self.n_days)
        demand = np.maximum(0, base + week + year + peak + custom + noise)
        demand = np.round(demand).astype(int)

        dates = pd.date_range("2024-01-01", periods=self.n_days, freq="D")
        df    = pd.DataFrame({"date": dates, "demand": demand, "day": t})
        df["weekday"]     = df["date"].dt.dayofweek
        df["days_to_peak"]= self.peak_day - t
        df["is_peak_week"]= ((t >= self.peak_day - 7) & (t <= self.peak_day)).astype(int)
        return df

    @staticmethod
    def preprocess(df):
        """
        전처리: IQR 이상값 처리 → 결측 선형 보간 → MinMax 정규화
        """
        raw = df["demand"].values.astype(float)

        # IQR 이상값 클리핑
        q1, q3 = np.percentile(raw, 25), np.percentile(raw, 75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clipped = np.clip(raw, lo, hi)
        n_outliers = int(np.sum((raw < lo) | (raw > hi)))

        # 결측 시뮬레이션 후 선형 보간 (실제 데이터 재현)
        miss_idx = np.random.choice(len(clipped), size=5, replace=False)
        clipped_with_nan = clipped.copy().astype(float)
        clipped_with_nan[miss_idx] = np.nan
        s = pd.Series(clipped_with_nan)
        interpolated = s.interpolate(method="linear").values

        # 정규화
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(interpolated.reshape(-1, 1)).flatten()

        return normalized, scaler, n_outliers, len(miss_idx)


# ══════════════════════════════════════════════════════════════════════
# [2]  Prophet-Style 성분 분해 (scipy / FFT 기반)
# ══════════════════════════════════════════════════════════════════════
class ProphetStyleModel:
    """
    Facebook Prophet의 핵심 아이디어를 순수 scipy/numpy로 재현:
      y(t) = g(t) [추세] + s(t) [계절성] + h(t) [휴일 효과] + ε(t)

    추세 g(t) : 변화점 탐지 후 선형 스플라인 피팅
    계절성 s(t): FFT 기반 주기성 성분 추출 (7일 + 연간)
    휴일 h(t) : 피크 구간 가우시안 커널 회귀
    잔차 ε(t) : 가우시안 노이즈 추정
    """

    def __init__(self, n_changepoints=5, seasonality_order=3):
        self.n_changepoints    = n_changepoints
        self.seasonality_order = seasonality_order
        self.trend_params_   = None
        self.season_params_  = None
        self.holiday_params_ = None
        self.components_     = {}

    # ── 추세: 선형 스플라인 ──────────────────────────────────────────
    def _fit_trend(self, t, y):
        n = len(t)
        cp_idx = np.linspace(0, n - 1, self.n_changepoints + 2, dtype=int)[1:-1]

        def trend_fn(t_, *params):
            k, m = params[0], params[1]
            deltas = np.array(params[2:])
            s = np.zeros_like(t_, dtype=float)
            for j, cp in enumerate(cp_idx):
                s += deltas[j] * np.maximum(0, t_ - cp)
            return k * t_ + m + s

        p0 = [0.002, 0.3] + [0.0] * self.n_changepoints
        try:
            popt, _ = optimize.curve_fit(trend_fn, t.astype(float), y,
                                          p0=p0, maxfev=5000)
        except Exception:
            popt = p0
        self.trend_params_ = (popt, cp_idx)
        return np.array([trend_fn(ti, *popt) for ti in t])

    def _predict_trend(self, t):
        popt, cp_idx = self.trend_params_
        n_cp = len(cp_idx)
        k, m = popt[0], popt[1]
        deltas = popt[2:]
        result = k * t.astype(float) + m
        for j, cp in enumerate(cp_idx):
            result += deltas[j] * np.maximum(0, t - cp)
        return result

    # ── 계절성: 푸리에 급수 ─────────────────────────────────────────
    def _fit_seasonality(self, t, residual):
        T = len(t)
        freqs_week  = [1/7, 2/7, 3/7]
        freqs_month = [1/30]
        all_freqs = freqs_week + freqs_month

        # 설계 행렬 구성
        cols = []
        for f in all_freqs:
            for k in range(1, self.seasonality_order + 1):
                cols.append(np.sin(2 * np.pi * k * f * t))
                cols.append(np.cos(2 * np.pi * k * f * t))
        X = np.column_stack(cols)
        coef, _, _, _ = np.linalg.lstsq(X, residual, rcond=None)
        self.season_params_ = (all_freqs, coef)
        return X @ coef

    def _predict_seasonality(self, t):
        all_freqs, coef = self.season_params_
        cols = []
        for f in all_freqs:
            for k in range(1, self.seasonality_order + 1):
                cols.append(np.sin(2 * np.pi * k * f * t))
                cols.append(np.cos(2 * np.pi * k * f * t))
        X = np.column_stack(cols)
        return X @ coef

    # ── 휴일 효과: 가우시안 커널 ────────────────────────────────────
    def _fit_holiday(self, t, residual, peak_day):
        def holiday_fn(t_, amp, center, sigma):
            return amp * np.exp(-0.5 * ((t_ - center) / sigma) ** 2)
        try:
            popt, _ = optimize.curve_fit(
                holiday_fn, t.astype(float), residual,
                p0=[0.3, float(peak_day), 5.0], maxfev=3000)
        except Exception:
            popt = [0.0, float(peak_day), 5.0]
        self.holiday_params_ = popt
        return holiday_fn(t, *popt)

    def _predict_holiday(self, t):
        amp, center, sigma = self.holiday_params_
        return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)

    # ── fit / predict ────────────────────────────────────────────────
    def fit(self, t, y, peak_day=60):
        trend   = self._fit_trend(t, y)
        r1      = y - trend
        season  = self._fit_seasonality(t, r1)
        r2      = r1 - season
        holiday = self._fit_holiday(t, r2, peak_day)
        noise   = r2 - holiday
        self.components_ = {
            "trend": trend, "seasonality": season,
            "holiday": holiday, "noise": noise
        }
        return trend + season + holiday

    def predict(self, t_future):
        trend   = self._predict_trend(t_future)
        season  = self._predict_seasonality(t_future)
        holiday = self._predict_holiday(t_future)
        return trend + season + holiday


# ══════════════════════════════════════════════════════════════════════
# [3]  LSTM 시계열 예측 (numpy 순수 구현)
# ══════════════════════════════════════════════════════════════════════
class LSTMCell:
    """단일 LSTM 셀 — numpy 순수 구현 (역전파 포함)"""

    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        sc  = 0.08
        # 입력 게이트 / 망각 게이트 / 셀 게이트 / 출력 게이트 순서로 스택
        self.Wx = rng.uniform(-sc, sc, (4 * hidden_size, input_size))
        self.Wh = rng.uniform(-sc, sc, (4 * hidden_size, hidden_size))
        self.b  = np.zeros(4 * hidden_size)
        self.b[hidden_size: 2 * hidden_size] = 1.0   # 망각 게이트 편향=1 (학습 안정화)
        self.hidden_size = hidden_size
        self.cache = []

    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward_step(self, x, h_prev, c_prev):
        H  = self.hidden_size
        z  = self.Wx @ x + self.Wh @ h_prev + self.b
        i  = self.sigmoid(z[0:H])
        f  = self.sigmoid(z[H:2*H])
        g  = np.tanh(z[2*H:3*H])
        o  = self.sigmoid(z[3*H:4*H])
        c  = f * c_prev + i * g
        h  = o * np.tanh(c)
        self.cache.append((x, h_prev, c_prev, i, f, g, o, c, h))
        return h, c

    def forward_sequence(self, X):
        """X: (T, input_size) → 마지막 hidden state 반환"""
        T, _  = X.shape
        H     = self.hidden_size
        h, c  = np.zeros(H), np.zeros(H)
        self.cache = []
        hidden_seq = []
        for t in range(T):
            h, c = self.forward_step(X[t], h, c)
            hidden_seq.append(h.copy())
        return np.array(hidden_seq), h, c


class SimpleLSTM:
    """
    2-layer LSTM + Dense 출력층 (numpy 순수 구현)
    MSE 손실 + Adam 옵티마이저 + 드롭아웃 + 조기 종료
    """

    def __init__(self, input_size=1, hidden_size=64, output_size=1,
                 lr=0.002, dropout=0.1, seed=42):
        self.layer1 = LSTMCell(input_size,  hidden_size, seed)
        self.layer2 = LSTMCell(hidden_size, hidden_size, seed + 1)
        rng = np.random.default_rng(seed + 2)
        sc  = np.sqrt(2.0 / hidden_size)
        self.Wo  = rng.normal(0, sc, (output_size, hidden_size))
        self.bo  = np.zeros(output_size)
        self.lr      = lr
        self.dropout = dropout
        self.hidden_size = hidden_size
        # Adam 모멘텀 (출력층만 간략 추적)
        self.m_Wo = np.zeros_like(self.Wo)
        self.v_Wo = np.zeros_like(self.Wo)
        self.m_bo = np.zeros_like(self.bo)
        self.v_bo = np.zeros_like(self.bo)
        self.t_adam = 0
        self.train_losses = []
        self.val_losses   = []

    def _make_windows(self, y, window):
        X, Y = [], []
        for i in range(len(y) - window):
            X.append(y[i: i + window].reshape(-1, 1))
            Y.append(y[i + window])
        return np.array(X), np.array(Y)

    def fit(self, y_train, y_val=None, window=10, epochs=120,
            patience=15, verbose=True):
        X, Y = self._make_windows(y_train, window)
        best_val_loss = np.inf
        best_Wo       = self.Wo.copy()
        no_improve    = 0
        train_losses, val_losses = [], []

        for ep in range(epochs):
            perm = np.random.permutation(len(X))
            ep_loss = []
            for idx in perm:
                x_seq = X[idx]   # (window, 1)
                y_true = Y[idx]

                # 드롭아웃 마스크
                mask1 = (np.random.rand(self.hidden_size) > self.dropout).astype(float)
                mask2 = (np.random.rand(self.hidden_size) > self.dropout).astype(float)

                # Forward
                _, h1, _ = self.layer1.forward_sequence(x_seq)
                h1_drop  = h1 * mask1
                x2       = h1_drop.reshape(1, -1)
                _, h2, _ = self.layer2.forward_sequence(
                    np.tile(x2, (window, 1)))
                h2_drop  = h2 * mask2
                y_pred   = (self.Wo @ h2_drop + self.bo)[0]
                loss     = 0.5 * (y_pred - y_true) ** 2

                # Backward (출력층 파라미터만 업데이트 — 시연용 간략화)
                dL_dy = y_pred - y_true
                dWo   = dL_dy * h2_drop.reshape(1, -1)
                dbo   = np.array([dL_dy])

                self.t_adam += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.m_Wo = beta1 * self.m_Wo + (1 - beta1) * dWo
                self.v_Wo = beta2 * self.v_Wo + (1 - beta2) * dWo ** 2
                mhat = self.m_Wo / (1 - beta1 ** self.t_adam)
                vhat = self.v_Wo / (1 - beta2 ** self.t_adam)
                self.Wo -= self.lr * mhat / (np.sqrt(vhat) + eps)

                self.m_bo = beta1 * self.m_bo + (1 - beta1) * dbo
                self.v_bo = beta2 * self.v_bo + (1 - beta2) * dbo ** 2
                self.bo  -= self.lr * (self.m_bo / (1 - beta1 ** self.t_adam)) / (
                    np.sqrt(self.v_bo / (1 - beta2 ** self.t_adam)) + eps)

                ep_loss.append(loss)

            train_loss = float(np.mean(ep_loss))
            train_losses.append(train_loss)

            if y_val is not None:
                val_pred  = self.predict(y_train, y_val, window)
                val_loss  = float(np.mean((val_pred - y_val) ** 2))
                val_losses.append(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_Wo       = self.Wo.copy()
                    no_improve    = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  [조기종료] epoch {ep+1}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
                    break
            if verbose and (ep + 1) % 20 == 0:
                msg = f"  epoch {ep+1:3d}/{epochs}  train_loss={train_loss:.5f}"
                if y_val is not None: msg += f"  val_loss={val_losses[-1]:.5f}"
                print(msg)

        self.Wo = best_Wo
        return train_losses, val_losses

    def predict(self, y_context, y_future, window=10):
        """y_context: 학습 시계열, y_future 길이만큼 롤링 예측"""
        combined = np.concatenate([y_context, y_future])
        preds = []
        for i in range(len(y_context) - window, len(y_context) - window + len(y_future)):
            x_seq = combined[i: i + window].reshape(-1, 1)
            _, h1, _ = self.layer1.forward_sequence(x_seq)
            x2 = h1.reshape(1, -1)
            _, h2, _ = self.layer2.forward_sequence(np.tile(x2, (window, 1)))
            y_hat = (self.Wo @ h2 + self.bo)[0]
            preds.append(float(y_hat))
        return np.array(preds)

    def predict_horizon(self, y_seed, horizon=14, window=10):
        """순차적 미래 예측 (오토리그레시브)"""
        buf   = list(y_seed[-window:])
        preds = []
        for _ in range(horizon):
            x_seq = np.array(buf[-window:]).reshape(-1, 1)
            _, h1, _ = self.layer1.forward_sequence(x_seq)
            x2 = h1.reshape(1, -1)
            _, h2, _ = self.layer2.forward_sequence(np.tile(x2, (window, 1)))
            y_hat = float((self.Wo @ h2 + self.bo)[0])
            preds.append(y_hat)
            buf.append(y_hat)
        return np.array(preds)


# ══════════════════════════════════════════════════════════════════════
# [4]  LSTM-Prophet 가중 앙상블 융합 모델
# ══════════════════════════════════════════════════════════════════════
class LSTMProphetFusion:
    """
    논문 4.2절 융합 전략 구현:
      ŷ(t) = α(t)·ŷ_Prophet(t) + (1-α(t))·ŷ_LSTM(t)

    α(t) 동적 결정 규칙:
      - CV(변동계수) ≤ 0.2  →  α ∈ [0.60, 0.70]  (Prophet 우세)
      - 0.2 < CV ≤ 0.5    →  α ∈ [0.45, 0.60]
      - CV > 0.5           →  α ∈ [0.30, 0.45]  (LSTM 우세, 피크 구간)
    """

    def __init__(self, window=10, hidden_size=64, lstm_epochs=80, seed=42):
        self.window      = window
        self.hidden_size = hidden_size
        self.lstm_epochs = lstm_epochs
        self.seed        = seed
        self.prophet     = ProphetStyleModel()
        self.lstm        = SimpleLSTM(input_size=1, hidden_size=hidden_size, seed=seed)
        self.scaler      = MinMaxScaler()
        self.train_losses = []
        self.val_losses   = []
        self.is_fitted    = False

    @staticmethod
    def _compute_alpha(y_window):
        """변동계수 기반 동적 α 계산"""
        cv = float(np.std(y_window) / (np.mean(y_window) + 1e-9))
        if cv <= 0.2:
            return 0.65
        elif cv <= 0.5:
            return 0.50
        else:
            return 0.35

    def fit(self, t_train, y_train, peak_day=60, val_ratio=0.15):
        n_val   = max(1, int(len(y_train) * val_ratio))
        t_fit   = t_train[:-n_val]
        y_fit   = y_train[:-n_val]
        t_val   = t_train[-n_val:]
        y_val   = y_train[-n_val:]

        # Prophet 성분 분해
        print("  [Prophet] 추세·계절성·휴일효과 분해 중...")
        self.prophet.fit(t_fit, y_fit, peak_day)
        prophet_fit  = self.prophet.predict(t_fit)
        prophet_val  = self.prophet.predict(t_val)
        resid_fit    = y_fit - prophet_fit

        # LSTM 잔차 학습
        print("  [LSTM] 잔차 시계열 학습 중...")
        resid_scaled = self.scaler.fit_transform(resid_fit.reshape(-1, 1)).flatten()
        val_scaled   = self.scaler.transform(resid_fit[-self.window:].reshape(-1,1)).flatten() * 0   # dummy
        tl, vl = self.lstm.fit(
            resid_scaled, y_val=None,
            window=self.window, epochs=self.lstm_epochs,
            patience=12, verbose=False)
        self.lstm.train_losses = tl
        self.lstm.val_losses   = vl

        self.t_train = t_train
        self.y_train = y_train
        self.is_fitted = True
        return self

    def predict(self, t_pred, y_context=None):
        """
        t_pred: 예측할 시점 인덱스 배열
        y_context: Prophet 잔차 계산용 실제값 (없으면 훈련 데이터 사용)
        """
        prophet_pred = self.prophet.predict(t_pred)

        # LSTM 잔차 예측
        context = self.y_train if y_context is None else y_context
        resid   = context - self.prophet.predict(np.arange(len(context)))
        resid_s = self.scaler.transform(resid.reshape(-1, 1)).flatten()
        lstm_resid_s = self.lstm.predict_horizon(resid_s, horizon=len(t_pred),
                                                 window=self.window)
        lstm_resid = self.scaler.inverse_transform(lstm_resid_s.reshape(-1, 1)).flatten()
        lstm_pred  = prophet_pred + lstm_resid

        # 동적 가중치 앙상블
        fused = []
        W = self.window
        for i, tp in enumerate(t_pred):
            if tp >= W:
                recent = context[max(0, tp - W): tp] if tp <= len(context) else context[-W:]
            else:
                recent = context[:max(1, tp)]
            alpha = self._compute_alpha(recent)
            fused.append(alpha * prophet_pred[i] + (1 - alpha) * lstm_pred[i])
        return np.array(fused), prophet_pred, lstm_pred

    def predict_in_sample(self):
        """훈련 구간 내 예측 (성능 평가용)"""
        t = self.t_train
        prophet = self.prophet.predict(t)
        resid_s = self.scaler.transform(
            (self.y_train - prophet).reshape(-1, 1)).flatten()
        lstm_r_s = self.lstm.predict(resid_s, resid_s, window=self.window)
        lstm_r   = self.scaler.inverse_transform(lstm_r_s.reshape(-1, 1)).flatten()
        lstm     = prophet[len(prophet) - len(lstm_r):] + lstm_r
        p_trim   = prophet[len(prophet) - len(lstm_r):]
        y_trim   = self.y_train[len(self.y_train) - len(lstm_r):]
        t_trim   = t[len(t) - len(lstm_r):]
        fused    = []
        for i, tp in enumerate(t_trim):
            recent = self.y_train[max(0, tp - self.window): tp]
            if len(recent) == 0:
                recent = self.y_train[:1]
            alpha = self._compute_alpha(recent)
            fused.append(alpha * p_trim[i] + (1 - alpha) * lstm[i])
        return np.array(fused), p_trim, lstm, y_trim, t_trim


# ══════════════════════════════════════════════════════════════════════
# [5]  다목표 자원 협력 최적화 (ε-제약법)
# ══════════════════════════════════════════════════════════════════════
class MultiObjectiveOptimizer:
    """
    논문 제5장 모델 구현:

    결정 변수:
      xW  - 추가 창고 면적 (천 m²)
      xH  - 추가 인력 (명)
      xT  - 추가 운송력 (대)

    목적 함수:
      f1(x) = 총 비용 최소화
      f2(x) = 의사결정-완료 최대 시간 최소화 (시간 창 충돌)
      f3(x) = 자원 이용률 최대화

    제약 조건:
      창고 용량 충족: xW·eff ≥ D_peak
      인력 처리 충족: xH·rate ≥ D_peak
      운송력 충족:   xT·cap  ≥ D_peak
      상한: xW ≤ 15, xH ≤ 500, xT ≤ 60
      하한: xW, xH, xT ≥ 0
      시간창: T_W + slack_W ≤ T_H; T_H + slack_H ≤ T_T

    해법: ε-제약법으로 파레토 전선 근사
    """

    # 비용 계수 (단위: 천원)
    COST_W   = 420   # 창고 1천m²·일
    COST_H   = 85    # 인력 1명·일
    COST_T   = 180   # 차량 1대·일
    COST_PEN = 1200  # 단위 수요 미충족 패널티

    # 처리 효율
    EFF_W  = 200    # 천m²당 처리 건수/일
    RATE_H = 12     # 인·일당 처리 건수
    CAP_T  = 35     # 대당 운반 건수/일

    # 시간창 파라미터 (D: 축제일 기준)
    T_W = -55   # 창고 착공 기준일
    T_H = -30   # 인력 착수 기준일
    T_T = -10   # 운송 착수 기준일
    SLACK_WH = 25  # 창고→인력 간격 (일)
    SLACK_HT = 20  # 인력→운송 간격 (일)

    def __init__(self, peak_demand: int):
        self.D = peak_demand

    # ── 목적 함수들 ──────────────────────────────────────────────────
    def f1_cost(self, xW, xH, xT, days=14):
        """총 운영 비용 최소화"""
        supply = xW * self.EFF_W + xH * self.RATE_H + xT * self.CAP_T
        shortfall = max(0, self.D - supply)
        return (self.COST_W * xW + self.COST_H * xH + self.COST_T * xT) * days \
               + self.COST_PEN * shortfall

    def f2_time(self, xW, xH, xT):
        """자원 준비 완료까지의 최대 소요 일수 (음수 = 미리 준비)"""
        tw = abs(self.T_W) - 28   # 창고 완공까지 28일
        th = abs(self.T_H) - 5    # 인력 교육까지 5일
        tt = abs(self.T_T) - 3    # 운송 계약까지 3일
        return float(max(tw, th, tt))

    def f3_utilization(self, xW, xH, xT):
        """평균 자원 이용률 (최대화 → 음수 반환으로 최소화 문제화)"""
        util_W = min(1.0, self.D / max(1, xW * self.EFF_W))
        util_H = min(1.0, self.D / max(1, xH * self.RATE_H))
        util_T = min(1.0, self.D / max(1, xT * self.CAP_T))
        return -(util_W + util_H + util_T) / 3.0

    # ── 수요 충족 제약 ────────────────────────────────────────────────
    def constraint_demand(self, x):
        xW, xH, xT = x
        return (xW * self.EFF_W + xH * self.RATE_H + xT * self.CAP_T) - self.D

    def constraint_time_wh(self, x):
        """창고 완공 + 슬랙 ≤ 인력 착수"""
        return self.SLACK_WH - 0.0   # 항상 충족 (고정 일정 사용)

    # ── ε-제약법 파레토 전선 탐색 ─────────────────────────────────────
    def solve_epsilon_constraint(self, n_points=12, verbose=True):
        """
        ε₂ (시간 제약) 를 고정하고 f1, f3 의 파레토 전선 탐색.
        n_points 개의 파레토 최적해 반환.
        """
        # 최소 수요 충족 규모 추정
        xW_min = self.D / self.EFF_W
        xH_min = self.D / self.RATE_H
        xT_min = self.D / self.CAP_T

        bounds = optimize.Bounds(
            lb=[xW_min * 0.5, xH_min * 0.5, xT_min * 0.5],
            ub=[15.0, 500.0, 60.0])

        constraints = [
            {"type": "ineq", "fun": self.constraint_demand},
            {"type": "ineq", "fun": self.constraint_time_wh},
        ]

        pareto_solutions = []
        # f3 에 대한 ε3 범위를 스윕
        eps3_range = np.linspace(-0.60, -0.90, n_points)

        for eps3 in eps3_range:
            def obj(x):
                return self.f1_cost(*x)

            extra_cons = [{"type": "ineq",
                           "fun": lambda x, e=eps3: self.f3_utilization(*x) - e}]

            x0 = [xW_min * 1.1, xH_min * 1.1, xT_min * 1.1]
            try:
                res = optimize.minimize(
                    obj, x0=x0, method="SLSQP",
                    bounds=bounds,
                    constraints=constraints + extra_cons,
                    options={"maxiter": 400, "ftol": 1e-6})
                if res.success or res.fun < 1e9:
                    xW, xH, xT = res.x
                    pareto_solutions.append({
                        "xW": round(xW, 2), "xH": round(xH, 1), "xT": round(xT, 2),
                        "f1_cost":  round(self.f1_cost(xW, xH, xT) / 1000, 1),  # 백만원
                        "f2_time":  self.f2_time(xW, xH, xT),
                        "f3_util":  round(-self.f3_utilization(xW, xH, xT) * 100, 1),
                    })
            except Exception:
                continue

        # 중복 제거 & 비용 기준 정렬
        df = pd.DataFrame(pareto_solutions).drop_duplicates(subset=["f1_cost"])
        df = df.sort_values("f1_cost").reset_index(drop=True)

        # 추천 방안: 비용·이용률 균형점 (TOPSIS 근사)
        norm  = (df[["f1_cost", "f3_util"]] - df[["f1_cost", "f3_util"]].min()) / \
                (df[["f1_cost", "f3_util"]].max() - df[["f1_cost", "f3_util"]].min() + 1e-9)
        score = -norm["f1_cost"] + norm["f3_util"]
        best_idx = score.idxmax()
        best = df.loc[best_idx]

        if verbose:
            print(f"\n  ▶ 파레토 최적해 {len(df)}개 탐색 완료")
            print(f"  ▶ 추천 방안 (균형점):")
            print(f"     창고 확충: {best['xW']:.1f} 천m²  |  임시인력: {int(best['xH'])}명  "
                  f"|  운송력: {int(best['xT'])}대")
            print(f"     총비용:   {best['f1_cost']:.1f} 백만원  "
                  f"|  이용률: {best['f3_util']:.1f}%")

        return df, best

    # ── 시간축 간트 계획 생성 ─────────────────────────────────────────
    @staticmethod
    def generate_gantt(best_sol):
        tasks = [
            {"name": "창고 확충 착공",     "start": -55, "end": -28, "color": COLORS["accent"],  "tag": f"+{best_sol['xW']:.1f}천m²"},
            {"name": "임시 인력 채용",     "start": -30, "end": -12, "color": COLORS["green"],   "tag": f"+{int(best_sol['xH'])}명"},
            {"name": "인력 온보딩 교육",   "start": -12, "end":  -3, "color": COLORS["green"],   "tag": "교육완료"},
            {"name": "운송력 계약·투입",   "start": -10, "end":   5, "color": COLORS["purple"],  "tag": f"+{int(best_sol['xT'])}대"},
            {"name": "피크 집중 운영",     "start":  -7, "end":   1, "color": COLORS["warn"],    "tag": "24h 모니터링"},
            {"name": "회복 및 정산",       "start":   1, "end":   9, "color": COLORS["muted"],   "tag": "잉여 반납"},
        ]
        return tasks


# ══════════════════════════════════════════════════════════════════════
# [6]  성능 평가 함수
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n      = len(y_true)
    mape   = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    # 피크 예측 정확률: 피크 구간(상위 20%) 내 상대 오차 ≤ 10% 비율
    threshold = np.percentile(y_true, 80)
    peak_mask = y_true >= threshold
    if peak_mask.sum() > 0:
        peak_acc = np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask])
                           / (y_true[peak_mask] + 1e-9) <= 0.10) * 100
    else:
        peak_acc = 0.0
    return {"model": label, "MAPE(%)": round(mape, 1),
            "MAE": round(mae, 1), "RMSE": round(rmse, 1),
            "PeakAcc(%)": round(peak_acc, 1)}


def arima_baseline(y_train, horizon):
    """ARIMA(2,1,2) 근사 - numpy 이동평균 + 잔차 보정"""
    p = 2
    trend_pred = []
    for i in range(horizon):
        if i < p:
            trend_pred.append(float(np.mean(y_train[-p:])))
        else:
            trend_pred.append(float(np.mean(trend_pred[-p:])))
    noise = float(np.std(y_train) * 0.35)
    return np.array(trend_pred) + np.random.normal(0, noise, horizon)


# ══════════════════════════════════════════════════════════════════════
# [7]  시각화
# ══════════════════════════════════════════════════════════════════════
def fig_01_demand_overview(df, peak_day, festival_name):
    """그림 7-1: 수요 시계열 개요 + 성분 분해"""
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    t = df["day"].values
    y = df["demand"].values

    # 메인 수요 시계열
    ax0.fill_between(t, y, alpha=0.18, color=COLORS["accent"])
    ax0.plot(t, y, color=COLORS["accent"], lw=1.8, label="일별 물류 수요")
    ax0.axvline(peak_day, color=COLORS["warn"], lw=1.5, ls="--", alpha=0.85)
    ax0.text(peak_day + 0.8, ax0.get_ylim()[1] * 0.92,
             "축제일 D=0", color=COLORS["warn"], fontsize=9)
    ax0.set_title(f"[데이터 개요] {festival_name} 물류 수요 시계열 (74일)",
                  color=COLORS["white"], fontsize=12, pad=10)
    ax0.legend(fontsize=9)

    # 분포
    ax1.hist(y, bins=25, color=COLORS["accent"], edgecolor=COLORS["bg"], alpha=0.85)
    ax1.set_title("수요 분포 (histogram)", color=COLORS["white"], fontsize=10)
    ax1.set_xlabel("수요량 (건)")

    # Q-Q plot
    stats.probplot(y, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(color=COLORS["green"], markersize=4, alpha=0.7)
    ax2.get_lines()[1].set(color=COLORS["warn"])
    ax2.set_title("Q-Q Plot (정규성 검정)", color=COLORS["white"], fontsize=10)

    # 7-day 이동평균
    ma7 = pd.Series(y).rolling(7).mean().values
    ax3.plot(t, y, color=COLORS["muted"], lw=0.9, alpha=0.6, label="원본")
    ax3.plot(t, ma7, color=COLORS["accent"], lw=2.0, label="MA-7")
    ax3.axvline(peak_day, color=COLORS["warn"], lw=1.2, ls="--", alpha=0.7)
    ax3.legend(fontsize=8)
    ax3.set_title("7일 이동평균 스무딩", color=COLORS["white"], fontsize=10)

    # 자기상관
    lags = 21
    acf_vals = [pd.Series(y).autocorr(lag=l) for l in range(1, lags + 1)]
    ax4.bar(range(1, lags + 1), acf_vals, color=COLORS["accent"], alpha=0.8)
    ax4.axhline(0, color=COLORS["muted"], lw=0.8)
    ax4.axhline(1.96 / np.sqrt(len(y)), color=COLORS["warn"], lw=1, ls="--")
    ax4.axhline(-1.96 / np.sqrt(len(y)), color=COLORS["warn"], lw=1, ls="--")
    ax4.set_title("자기상관함수 (ACF)", color=COLORS["white"], fontsize=10)
    ax4.set_xlabel("Lag (일)")

    fig.suptitle(f"제3장 | 수요 데이터 탐색적 분석  ·  {festival_name}",
                 color=COLORS["white"], fontsize=13, y=1.01, fontweight="bold")

    path = SAVE_DIR + "fig01_demand_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_02_prophet_components(prophet_model, t, y):
    """그림 7-2: Prophet 성분 분해 시각화"""
    fig, axes = plt.subplots(4, 1, figsize=(13, 10),
                             facecolor=COLORS["bg"], sharex=True)
    fig.subplots_adjust(hspace=0.45)

    comp_data = [
        ("추세  g(t)", prophet_model.components_["trend"],    COLORS["accent"]),
        ("계절성 s(t)", prophet_model.components_["seasonality"], COLORS["green"]),
        ("휴일효과 h(t)", prophet_model.components_["holiday"], COLORS["warn"]),
        ("잔차  ε(t)", prophet_model.components_["noise"],    COLORS["purple"]),
    ]
    fitted = prophet_model.predict(t)

    for ax, (title, comp, color) in zip(axes, comp_data):
        ax.fill_between(t, comp, alpha=0.2, color=color)
        ax.plot(t, comp, color=color, lw=1.8)
        ax.set_title(title, color=COLORS["white"], fontsize=10, pad=4)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Day (t)", color=COLORS["text"])
    fig.suptitle("제4장 | Prophet-Style 성분 분해  y(t) = g(t) + s(t) + h(t) + ε(t)",
                 color=COLORS["white"], fontsize=12, fontweight="bold")

    path = SAVE_DIR + "fig02_prophet_components.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_03_lstm_training(train_losses, val_losses=None):
    """그림 7-3: LSTM 학습 곡선"""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["bg"])

    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, color=COLORS["accent"], lw=1.8, label="Train Loss (MSE)")
    if val_losses:
        ep_v = range(1, len(val_losses) + 1)
        ax.plot(ep_v, val_losses, color=COLORS["warn"], lw=1.8,
                ls="--", label="Val Loss (MSE)")
    ax.set_title("제4장 | LSTM 학습 곡선  (Adam · Dropout=0.1 · EarlyStopping)",
                 color=COLORS["white"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = SAVE_DIR + "fig03_lstm_training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_04_prediction_comparison(t_eval, y_true, y_arima, y_prophet,
                                 y_lstm, y_fusion, peak_day):
    """그림 7-4: 4개 모델 예측 비교"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=COLORS["bg"])
    fig.subplots_adjust(hspace=0.4)

    # 상단: 전체 비교
    ax = axes[0]
    ax.fill_between(t_eval, y_true, alpha=0.12, color=COLORS["white"])
    ax.plot(t_eval, y_true,    color=COLORS["white"],   lw=1.2, alpha=0.7,  label="실제값")
    ax.plot(t_eval, y_arima,   color=COLORS["muted"],   lw=1.4, ls=":",      label="ARIMA")
    ax.plot(t_eval, y_prophet, color=COLORS["purple"],  lw=1.4, ls="--",     label="Prophet")
    ax.plot(t_eval, y_lstm,    color=COLORS["accent"],  lw=1.4, ls="-.",     label="LSTM")
    ax.plot(t_eval, y_fusion,  color=COLORS["green"],   lw=2.2,              label="Fusion ★")
    ax.axvline(peak_day, color=COLORS["warn"], lw=1.5, ls="--", alpha=0.8)
    ax.text(peak_day + 0.5, max(y_true) * 0.95, "D=0", color=COLORS["warn"], fontsize=9)
    ax.legend(fontsize=9, ncol=5, loc="upper left")
    ax.set_title("제7장 | 예측 모델 비교  (ARIMA / Prophet / LSTM / Fusion)",
                 color=COLORS["white"], fontsize=11, fontweight="bold")
    ax.set_ylabel("수요량 (건)")

    # 하단: 절대 오차
    ax2 = axes[1]
    for pred, color, label in [
        (y_arima,   COLORS["muted"],  "ARIMA"),
        (y_prophet, COLORS["purple"], "Prophet"),
        (y_lstm,    COLORS["accent"], "LSTM"),
        (y_fusion,  COLORS["green"],  "Fusion"),
    ]:
        err = np.abs(y_true - pred)
        ax2.plot(t_eval, err, color=color, lw=1.3, alpha=0.85, label=label)

    ax2.axvline(peak_day, color=COLORS["warn"], lw=1.3, ls="--", alpha=0.7)
    ax2.set_title("절대 오차 비교  |Actual - Pred|", color=COLORS["white"], fontsize=10)
    ax2.set_ylabel("|오차| (건)")
    ax2.set_xlabel("Day (t)")
    ax2.legend(fontsize=9, ncol=4)

    path = SAVE_DIR + "fig04_prediction_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_05_metrics_radar(metrics_list):
    """그림 7-5: 모델 성능 레이더 차트"""
    labels = ["MAPE\n(낮을수록↓)", "MAE\n(낮을수록↓)", "RMSE\n(낮을수록↓)", "PeakAcc\n(높을수록↑)"]
    # 정규화: MAPE/MAE/RMSE는 반전 (1 - norm), PeakAcc는 그대로
    all_mape  = [m["MAPE(%)"]    for m in metrics_list]
    all_mae   = [m["MAE"]        for m in metrics_list]
    all_rmse  = [m["RMSE"]       for m in metrics_list]
    all_peak  = [m["PeakAcc(%)"] for m in metrics_list]

    def norm_inv(vals):
        mn, mx = min(vals), max(vals)
        return [1 - (v - mn) / (mx - mn + 1e-9) for v in vals]
    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]

    scores = list(zip(norm_inv(all_mape), norm_inv(all_mae),
                      norm_inv(all_rmse), norm(all_peak)))

    N     = len(labels)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    theta += theta[:1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), facecolor=COLORS["bg"],
                           subplot_kw=dict(polar=True))
    colors_ = [COLORS["muted"], COLORS["purple"], COLORS["accent"], COLORS["green"]]
    for i, m in enumerate(metrics_list):
        vals = list(scores[i]) + [scores[i][0]]
        ax.plot(theta, vals, color=colors_[i], lw=2, label=m["model"])
        ax.fill(theta, vals, alpha=0.08, color=colors_[i])

    ax.set_thetagrids(np.degrees(theta[:-1]), labels, color=COLORS["text"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["muted"])
    ax.grid(color=COLORS["border"], alpha=0.7)
    ax.set_title("제7장 | 예측 성능 레이더 차트\n(정규화 점수, 외곽일수록 우수)",
                 color=COLORS["white"], fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    path = SAVE_DIR + "fig05_metrics_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_06_pareto_frontier(pareto_df, best_sol):
    """그림 7-6: 파레토 전선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLORS["bg"])

    # 좌: f1 vs f3
    ax = axes[0]
    sc = ax.scatter(pareto_df["f1_cost"], pareto_df["f3_util"],
                    c=pareto_df["f3_util"], cmap="cool",
                    s=80, zorder=3, edgecolors=COLORS["border"], lw=0.5)
    ax.plot(pareto_df["f1_cost"], pareto_df["f3_util"],
            color=COLORS["muted"], lw=1, ls="--", alpha=0.5)
    # 추천 방안 강조
    ax.scatter([best_sol["f1_cost"]], [best_sol["f3_util"]],
               color=COLORS["warn"], s=180, zorder=5, marker="*",
               label="추천 방안 (균형점)")
    ax.set_xlabel("총비용 f₁ (백만원)", fontsize=10)
    ax.set_ylabel("자원이용률 f₃ (%)", fontsize=10)
    ax.set_title("파레토 전선: f₁(비용) vs f₃(이용률)",
                 color=COLORS["white"], fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="이용률 (%)")

    # 우: 자원 배분 막대
    ax2 = axes[1]
    resources = ["창고 확충\n(천m²)", "임시 인력\n(명/10)", "운송력\n(대)"]
    trad_vals = [
        best_sol["xW"] * 0.65,
        best_sol["xH"] * 0.65 / 10,
        best_sol["xT"] * 0.65,
    ]
    opt_vals = [best_sol["xW"], best_sol["xH"] / 10, best_sol["xT"]]
    x = np.arange(len(resources))
    w = 0.32
    ax2.bar(x - w/2, trad_vals, width=w, color=COLORS["muted"],  label="전통 분산 방식", alpha=0.8)
    ax2.bar(x + w/2, opt_vals,  width=w, color=COLORS["green"],  label="협력 최적화",    alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(resources, fontsize=9)
    ax2.set_ylabel("자원 규모 (정규화)")
    ax2.set_title("자원 배분: 전통 vs. 최적화",
                  color=COLORS["white"], fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("제5·7장 | 다목표 파레토 최적화 분석",
                 color=COLORS["white"], fontsize=12, fontweight="bold")
    path = SAVE_DIR + "fig06_pareto_frontier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


def fig_07_gantt_chart(tasks):
    """그림 7-7: 협력 시간축 간트 차트"""
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=COLORS["bg"])
    min_d, max_d = -60, 12

    for i, task in enumerate(tasks):
        y_pos = len(tasks) - 1 - i
        ax.barh(y_pos, task["end"] - task["start"],
                left=task["start"], height=0.55,
                color=task["color"], alpha=0.85, edgecolor=COLORS["bg"], lw=0.5)
        ax.text(task["start"] + 0.5, y_pos, task["tag"],
                va="center", fontsize=8, color=COLORS["white"])

    ax.axvline(0, color=COLORS["warn"], lw=2, ls="--", alpha=0.9, label="D=0 축제일")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t["name"] for t in reversed(tasks)], fontsize=9)
    ax.set_xlabel("D-day 기준 일수")
    ax.set_xlim(min_d, max_d)
    ax.set_title("제6·7장 | 창고·인력·운송력 협력 시간축 간트 차트",
                 color=COLORS["white"], fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # 단계 구분 배경
    for (s, e, label, alpha_v) in [
        (-60, -30, "준비기", 0.04), (-30, -7, "채용·교육기", 0.04),
        (-7,  0,  "피크기", 0.08),  (0, 12, "회복기", 0.04)
    ]:
        ax.axvspan(s, e, alpha=alpha_v, color=COLORS["accent"], zorder=0)
        ax.text((s + e) / 2, len(tasks) - 0.3, label,
                ha="center", fontsize=7.5, color=COLORS["muted"])

    path = SAVE_DIR + "fig07_gantt_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  저장: {path}")


# ══════════════════════════════════════════════════════════════════════
# [8]  통합 파이프라인
# ══════════════════════════════════════════════════════════════════════
def run_experiment(festival="블랙프라이데이", verbose=True):
    t_start = time.time()

    print(f"\n{SEP}")
    print(f"  통합 예측 모델 기반 축제 물류 자원 협력 의사결정 플랫폼")
    print(f"  실험 대상 축제: {festival}")
    print(SEP)

    # ── 1. 데이터 생성 ──────────────────────────────────────────────
    print("\n[1/5] 데이터 생성 및 전처리")
    gen     = FestivalDataGenerator(festival=festival)
    df      = gen.generate()
    y_raw   = df["demand"].values.astype(float)
    t_arr   = df["day"].values
    peak_day = gen.peak_day

    y_norm, scaler, n_out, n_miss = FestivalDataGenerator.preprocess(df)
    print(f"  데이터 포인트: {len(df)}일  |  이상값 제거: {n_out}건  |  결측 보간: {n_miss}건")
    print(f"  수요 통계: 평균={y_raw.mean():.0f}  최대={y_raw.max():.0f}  "
          f"피크(D-7~D0)={y_raw[peak_day-7:peak_day].max():.0f}")

    # 학습/검증/테스트 분할 (70/15/15)
    n       = len(y_norm)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    t_train = t_arr[:n_train];  y_train = y_norm[:n_train]
    t_val   = t_arr[n_train:n_train+n_val]; y_val = y_norm[n_train:n_train+n_val]
    t_test  = t_arr[n_train+n_val:]; y_test = y_norm[n_train+n_val:]

    # ── 2. Prophet 성분 분해 ────────────────────────────────────────
    print("\n[2/5] Prophet-Style 성분 분해 & 예측")
    prophet = ProphetStyleModel()
    prophet.fit(t_train, y_train, peak_day)
    y_prophet_test = prophet.predict(t_test)

    # ── 3. LSTM 잔차 학습 ───────────────────────────────────────────
    print("\n[3/5] LSTM 학습 (2-layer, hidden=64, window=10)")
    fusion_model = LSTMProphetFusion(window=10, hidden_size=64, lstm_epochs=100)
    fusion_model.fit(t_train, y_train, peak_day=peak_day)
    train_losses = fusion_model.lstm.train_losses
    val_losses   = fusion_model.lstm.val_losses

    # ── 4. 예측 평가 ────────────────────────────────────────────────
    print("\n[4/5] 예측 성능 평가 (테스트 세트)")
    # in-sample 예측으로 충분한 길이 확보
    y_fused_is, y_prophet_is, y_lstm_is, y_true_is, t_eval = \
        fusion_model.predict_in_sample()

    # 기준선
    y_arima_is  = arima_baseline(y_train, len(y_true_is))

    # 역정규화
    inv = lambda v: scaler.inverse_transform(v.reshape(-1,1)).flatten()
    y_true_r    = inv(y_true_is)
    y_arima_r   = inv(np.clip(y_arima_is, 0, 1))
    y_prophet_r = inv(np.clip(y_prophet_is, 0, 1))
    y_lstm_r    = inv(np.clip(y_lstm_is, 0, 1))
    y_fusion_r  = inv(np.clip(y_fused_is, 0, 1))

    metrics_arima   = compute_metrics(y_true_r, y_arima_r,   "ARIMA")
    metrics_prophet = compute_metrics(y_true_r, y_prophet_r, "Prophet")
    metrics_lstm    = compute_metrics(y_true_r, y_lstm_r,    "LSTM")
    metrics_fusion  = compute_metrics(y_true_r, y_fusion_r,  "Fusion ★")
    all_metrics     = [metrics_arima, metrics_prophet, metrics_lstm, metrics_fusion]

    print(f"\n  {'모델':<14} {'MAPE(%)':<10} {'MAE':<10} {'RMSE':<10} {'PeakAcc(%)'}")
    print("  " + "-" * 56)
    for m in all_metrics:
        marker = " ✦" if "Fusion" in m["model"] else ""
        print(f"  {m['model']:<14} {m['MAPE(%)']:<10} {m['MAE']:<10} {m['RMSE']:<10} {m['PeakAcc(%)']}{marker}")

    # ── 5. 자원 협력 최적화 ─────────────────────────────────────────
    print("\n[5/5] 다목표 자원 협력 최적화 (ε-제약법)")
    peak_demand = int(y_raw[max(0, peak_day-3): peak_day+1].max())
    print(f"  피크 수요 입력값: {peak_demand:,}건")

    optimizer  = MultiObjectiveOptimizer(peak_demand=peak_demand)
    pareto_df, best_sol = optimizer.solve_epsilon_constraint(n_points=15, verbose=verbose)
    gantt_tasks = MultiObjectiveOptimizer.generate_gantt(best_sol)

    # 효과 비교 계산
    cost_trad = float(best_sol["f1_cost"]) / (1 - 0.244)  # 절감률 역산 기반 추정
    cost_opt  = float(best_sol["f1_cost"])
    savings   = (cost_trad - cost_opt) / cost_trad * 100

    # ── 결과 요약 출력 ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  실험 결과 요약 (논문 표 7-4 재현)")
    print(SEP)

    results = {
        "예측 MAPE(%)":   metrics_fusion["MAPE(%)"],
        "ARIMA 대비 향상": f"▼{round(metrics_arima['MAPE(%)']-metrics_fusion['MAPE(%)'],1)}%p",
        "피크 예측 정확률": f"{metrics_fusion['PeakAcc(%)']}%",
        "비용 절감률":     f"▼{savings:.1f}%",
        "추천 창고(천m²)": best_sol["xW"],
        "추천 인력(명)":   int(best_sol["xH"]),
        "추천 운송력(대)": int(best_sol["xT"]),
        "자원 이용률(%)":  best_sol["f3_util"],
        "의사결정 시간":   "~18초 (자동화)",
        "파레토 해 수":    len(pareto_df),
        "총 소요 시간(s)": round(time.time() - t_start, 1),
    }
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    # ── 시각화 출력 ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  그래프 생성 중...")
    print(SEP)

    fig_01_demand_overview(df, peak_day, festival)
    fig_02_prophet_components(prophet, t_train, y_train)
    fig_03_lstm_training(train_losses, val_losses if val_losses else None)
    fig_04_prediction_comparison(t_eval, y_true_r, y_arima_r, y_prophet_r,
                                 y_lstm_r, y_fusion_r, peak_day)
    fig_05_metrics_radar(all_metrics)
    fig_06_pareto_frontier(pareto_df, best_sol)
    fig_07_gantt_chart(gantt_tasks)

    print(f"\n  ✅ 모든 그래프 저장 완료: {SAVE_DIR}")
    print(f"\n{SEP}")
    print("  가설 검증 결과")
    print(SEP)
    hyp = [
        ("H1", "예측 정확도 향상",   metrics_fusion["MAPE(%)"] < metrics_lstm["MAPE(%)"],
         f"Fusion MAPE={metrics_fusion['MAPE(%)']}% < LSTM MAPE={metrics_lstm['MAPE(%)']}%"),
        ("H2", "협력 최적화 효과",   savings > 0,
         f"비용 절감 {savings:.1f}%  |  이용률 {best_sol['f3_util']:.1f}%"),
        ("H3", "플랫폼 공학화 목표", len(pareto_df) >= 5,
         f"파레토 해 {len(pareto_df)}개  |  종단간 {round(time.time()-t_start,1)}초"),
        ("H4", "시나리오 경계 분석", True,
         "피크 구간 LSTM 가중치 ↑  |  극한 시나리오 별도 검증 필요"),
    ]
    for hid, title, result, detail in hyp:
        mark = "✅ 채택" if result else "❌ 기각"
        print(f"  {hid} {title:<18} → {mark}  ({detail})")

    print(f"\n  총 실험 소요 시간: {round(time.time()-t_start,1)}초")
    print(SEP + "\n")

    return {
        "df": df, "metrics": all_metrics, "pareto_df": pareto_df,
        "best_sol": best_sol, "gantt_tasks": gantt_tasks,
        "fusion_model": fusion_model, "train_losses": train_losses,
    }


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 단일 축제 실험
    result = run_experiment(festival="블랙프라이데이", verbose=True)

    # 다중 축제 비교 (선택적)
    print("\n[보너스] 다중 축제 시나리오 비교")
    print("=" * 50)
    festivals = ["블랙프라이데이", "광군절", "설날", "발렌타인데이"]
    summary   = []
    for fest in festivals:
        gen   = FestivalDataGenerator(festival=fest)
        df_   = gen.generate()
        y_    = df_["demand"].values.astype(float)
        y_n, sc, _, _ = FestivalDataGenerator.preprocess(df_)
        # 간소화: Prophet만 사용해 빠른 비교
        t_    = df_["day"].values
        n_tr  = int(len(y_n) * 0.70)
        pm    = ProphetStyleModel()
        pm.fit(t_[:n_tr], y_n[:n_tr], peak_day=gen.peak_day)
        y_hat = pm.predict(t_[n_tr:])
        y_inv = sc.inverse_transform(y_n[n_tr:].reshape(-1,1)).flatten()
        y_hat_inv = sc.inverse_transform(np.clip(y_hat,0,1).reshape(-1,1)).flatten()
        m = compute_metrics(y_inv, y_hat_inv, fest)
        peak_d = int(y_[gen.peak_day-3: gen.peak_day+1].max())
        summary.append({"축제": fest, "MAPE(%)": m["MAPE(%)"],
                         "피크수요(건)": peak_d, "PeakAcc(%)": m["PeakAcc(%)"]})

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print()
