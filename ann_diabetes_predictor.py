# -*- coding: utf-8 -*-
"""
인공신경망(다층 퍼셉트론, MLP)으로 당뇨병 진행도를 예측하는 프로그램

데이터
  scikit-learn 공개 데이터셋 'Diabetes' (Efron et al., 2004) - data/diabetes.csv
  환자 442명, 입력 특성 10개, 목표값 = 1년 뒤 질병 진행 정도 (25~346)

모델
  입력 10 → 은닉 32 (ReLU) → 은닉 16 (ReLU) → 출력 1 (선형)
  딥러닝 라이브러리 없이 NumPy만으로 순전파 · 역전파 · Adam 최적화를 직접 구현
  비교 기준: 평균값 예측, 선형회귀

사용법
  python ann_diabetes_predictor.py                 # 학습 + 결과 출력 + 그래프(images/) + model.json 저장
  python ann_diabetes_predictor.py --show          # 그래프 창 띄우기
  python ann_diabetes_predictor.py --predict 50 1 27.5 95 190 115 50 4 4.6 90
                                                   # 나이 성별 BMI 혈압 s1~s6 순서로 넣으면 진행도 예측
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
IMAGE_DIR = ROOT / "images"
MODEL_PATH = ROOT / "model.json"

SEED = 42
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
FEATURE_KO = ["나이", "성별", "BMI", "평균 혈압", "총콜레스테롤", "LDL",
              "HDL", "TC/HDL", "중성지방(로그)", "혈당"]

# 하이퍼파라미터
LAYERS = [10, 32, 16, 1]
EPOCHS, BATCH, LR, L2, PATIENCE = 1000, 32, 1e-3, 1e-3, 50

rng = np.random.default_rng(SEED)


# ── 데이터 ───────────────────────────────────────────
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["progression"].to_numpy(dtype=float).reshape(-1, 1)
    return X, y


def split_data(X, y):
    """학습 70% / 검증 15% / 테스트 15%"""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


class Scaler:
    """표준화: 학습 데이터의 평균·표준편차만 사용 (테스트 정보 누출 방지)"""

    def fit(self, a):
        self.mean, self.std = a.mean(0), a.std(0)
        return self

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


# ── 신경망 ───────────────────────────────────────────
class MLP:
    def __init__(self, sizes, l2=L2):
        self.l2 = l2
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He 초기화: ReLU 층에서 신호 분산이 유지되도록 표준편차 sqrt(2/n_in)
            self.W.append(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        # Adam 상태값 (1차·2차 모멘트)
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        """순전파: z = aW + b, 은닉층은 a = ReLU(z), 출력층은 선형"""
        self.cache = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = z if i == len(self.W) - 1 else np.maximum(0, z)
            self.cache.append(a)
        return a

    def backward(self, y):
        """역전파: 손실(MSE + L2)의 기울기를 연쇄법칙으로 출력층부터 계산"""
        n = y.shape[0]
        dA = 2 * (self.cache[-1] - y) / n            # ∂MSE/∂ŷ
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i != len(self.W) - 1:                  # ReLU 미분: z>0 → 1, 아니면 0
                dA = dA * (self.cache[i + 1] > 0)
            gW[i] = self.cache[i].T @ dA + self.l2 * self.W[i]
            gb[i] = dA.sum(0, keepdims=True)
            dA = dA @ self.W[i].T
        return gW, gb

    def step(self, gW, gb, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        """Adam: 기울기의 이동평균(관성)과 제곱 이동평균(크기)으로 파라미터별 학습률 조절"""
        self.t += 1
        for i in range(len(self.W)):
            for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                M[i] = b1 * M[i] + (1 - b1) * G[i]
                V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                m_hat = M[i] / (1 - b1 ** self.t)
                v_hat = V[i] / (1 - b2 ** self.t)
                P[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def get_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, params):
        self.W, self.b = [w.copy() for w in params[0]], [b.copy() for b in params[1]]


def train(Xtr, ytr, Xva, yva):
    """미니배치 학습 + 조기 종료 (검증 손실이 PATIENCE 에폭 동안 개선 없으면 중단)"""
    model = MLP(LAYERS)
    hist = {"train": [], "val": []}
    best_val, best_params, best_epoch, wait = np.inf, None, 0, 0

    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(len(Xtr))               # 매 에폭마다 순서 섞기
        for s in range(0, len(idx), BATCH):
            bi = idx[s:s + BATCH]
            model.forward(Xtr[bi])
            model.step(*model.backward(ytr[bi]))

        hist["train"].append(float(np.mean((model.forward(Xtr) - ytr) ** 2)))
        hist["val"].append(float(np.mean((model.forward(Xva) - yva) ** 2)))

        if hist["val"][-1] < best_val:                 # 최저 검증 손실일 때 가중치 저장
            best_val, best_params, best_epoch, wait = hist["val"][-1], model.get_params(), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.set_params(best_params)
    return model, hist, best_epoch, best_val


# ── 평가 지표 ─────────────────────────────────────────
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2_score(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


# ── 모델 저장 / 불러오기 ───────────────────────────────
def save_model(model, xs, ys, lin, path=MODEL_PATH):
    """웹 계산기(index.html)와 --predict에서 쓰는 가중치 파일"""
    data = {
        "features": FEATURES,
        "layers": LAYERS,
        "x_mean": xs.mean.tolist(), "x_std": xs.std.tolist(),
        "y_mean": float(ys.mean[0]), "y_std": float(ys.std[0]),
        "W": [w.round(8).tolist() for w in model.W],
        "b": [b.ravel().round(8).tolist() for b in model.b],
        "linear": {"coef": lin.coef_.ravel().round(8).tolist(), "intercept": float(lin.intercept_[0])},
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def predict_from_file(values, path=MODEL_PATH):
    m = json.loads(path.read_text(encoding="utf-8"))
    a = (np.asarray(values, float) - np.array(m["x_mean"])) / np.array(m["x_std"])
    for i, (W, b) in enumerate(zip(m["W"], m["b"])):
        a = a @ np.array(W) + np.array(b)
        if i < len(m["W"]) - 1:
            a = np.maximum(0, a)
    ann = float(a[0] * m["y_std"] + m["y_mean"])
    lin = float(np.dot((np.asarray(values, float) - m["x_mean"]) / m["x_std"], m["linear"]["coef"])
                + m["linear"]["intercept"])
    return ann, lin


# ── 그래프 ────────────────────────────────────────────
def set_korean_font(plt):
    """운영체제에 설치된 한글 폰트를 자동 선택 (Windows / macOS / Linux)"""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo",
                 "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_plots(hist, best_epoch, y_test, pred_ann, pred_lin, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_korean_font(plt)
    IMAGE_DIR.mkdir(exist_ok=True)

    # 1) 학습 곡선
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(range(1, len(hist["train"]) + 1), hist["train"], lw=2, color="#2563eb", label="학습 손실")
    ax.plot(range(1, len(hist["val"]) + 1), hist["val"], lw=2, color="#ea580c", label="검증 손실")
    ax.axvline(best_epoch, ls="--", color="gray", lw=1, label=f"최적 에폭 ({best_epoch})")
    ax.set(title="학습 곡선 (표준화된 MSE)", xlabel="에폭", ylabel="MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "learning_curve.png", dpi=150)

    # 2) 실제값 vs 예측값 (선형회귀와 비교)
    fig2, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, pred, name, color in ((axes[0], pred_lin, "선형회귀", "#6b7280"),
                                  (axes[1], pred_ann, "인공신경망", "#2563eb")):
        ax.scatter(y_test, pred, s=28, alpha=0.75, color=color, edgecolor="white")
        ax.plot([0, 360], [0, 360], ls="--", color="gray", lw=1, label="완벽한 예측 (y = x)")
        ax.set(title=f"{name}  (R² = {r2_score(y_test, pred):.3f})", xlabel="실제 질병 진행도",
               xlim=(0, 360), ylim=(0, 360))
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("예측 질병 진행도")
    fig2.suptitle("테스트 데이터: 실제값 vs 예측값")
    fig2.tight_layout()
    fig2.savefig(IMAGE_DIR / "prediction.png", dpi=150)

    if show:
        plt.show()


# ── 실행 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="인공신경망으로 당뇨병 진행도 예측")
    parser.add_argument("--predict", type=float, nargs=10, metavar="V",
                        help="나이 성별 BMI 혈압 s1 s2 s3 s4 s5 s6 (model.json 필요)")
    parser.add_argument("--show", action="store_true", help="그래프 창 띄우기")
    args = parser.parse_args()

    if args.predict is not None:
        if not MODEL_PATH.exists():
            parser.error("model.json이 없습니다. 먼저 인자 없이 실행해 학습하세요.")
        ann, lin = predict_from_file(args.predict)
        print("입력값")
        for ko, v in zip(FEATURE_KO, args.predict):
            print(f"  {ko:<10} {v:g}")
        print(f"\n예측 질병 진행도  인공신경망 {ann:.1f}  /  선형회귀 {lin:.1f}")
        return

    X, y = load_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
    xs, ys = Scaler().fit(X_tr), Scaler().fit(y_tr)
    Xtr, Xva, Xte = xs.transform(X_tr), xs.transform(X_va), xs.transform(X_te)

    print(f"데이터: {len(X)}명  (학습 {len(X_tr)} / 검증 {len(X_va)} / 테스트 {len(X_te)})")
    model, hist, best_epoch, best_val = train(Xtr, ys.transform(y_tr), Xva, ys.transform(y_va))
    print(f"학습 종료: {len(hist['train'])} 에폭 (최적 에폭 {best_epoch}, 검증 MSE {best_val:.4f})")

    pred_ann = ys.inverse(model.forward(Xte))
    lin = LinearRegression().fit(Xtr, y_tr)
    pred_lin = lin.predict(Xte)
    pred_mean = np.full_like(y_te, y_tr.mean())

    print(f"\n[테스트 성능]  n = {len(y_te)}")
    print(f"{'모델':<12}{'MAE':>8}{'RMSE':>8}{'R²':>8}")
    for name, p in (("평균값 예측", pred_mean), ("선형회귀", pred_lin), ("인공신경망", pred_ann)):
        print(f"{name:<12}{mae(y_te, p):8.2f}{rmse(y_te, p):8.2f}{r2_score(y_te, p):8.3f}")

    print("\n[예측 예시: 테스트 환자 5명]")
    for i in range(5):
        print(f"  환자 {i + 1}: 실제 {y_te[i, 0]:6.1f}  →  신경망 {pred_ann[i, 0]:6.1f}, 선형 {pred_lin[i, 0]:6.1f}")

    save_model(model, xs, ys, lin)
    save_plots(hist, best_epoch, y_te, pred_ann, pred_lin, show=args.show)
    print(f"\n저장: {MODEL_PATH.name}, images/learning_curve.png, images/prediction.png")


if __name__ == "__main__":
    main()
