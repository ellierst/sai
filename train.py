"""
Лабораторна робота №3 — Варіант 11
Навчання нейронної мережі
Символи: L, м, Q, M
Активація: sigmoid (шар 1), tanh (шар 2)
Алгоритм: L-BFGS (аналог trainbfg у MATLAB)
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from scipy.optimize import minimize

# ══════════════════════════════════════════
# НАЛАШТУВАННЯ — змінюйте тільки тут
# ══════════════════════════════════════════
TRAIN_FOLDER  = "Neuro_Train"
FILE_NAME_TX  = "Neuro_Train/File_Name_tx1.txt"
CLASS_NAME_TY = "Neuro_Train/Class_Name_ty1.txt"

MSE_GOAL         = 1e-4   # змінюйте: 1e-2, 1e-3, 1e-4
MAX_EPOCHS       = 1500
MAX_TIME         = 120    # секунд

LAYER1_NEURONS   = 16     # нейронів у шарі 1
LAYER2_NEURONS   = 12     # нейронів у шарі 2
# ══════════════════════════════════════════


# ─── Активаційні функції ───────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_d(x):
    return 1.0 - np.tanh(x) ** 2


# ─── Зчитування зображень ─────────────────
def load_images(file_name_tx, train_folder, num_classes):
    P, T = [], []
    with open(file_name_tx, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts   = line.split()
        cls     = int(parts[0])
        name    = parts[1]

        path = os.path.join(train_folder, name + ".bmp")
        img  = Image.open(path).convert("L").resize((16, 16))
        vec  = np.array(img, dtype=np.float64).flatten() / 255.0  # 256 значень

        P.append(vec)

        t = np.zeros(num_classes)
        t[cls - 1] = 1.0
        T.append(t)

    return np.array(P).T, np.array(T).T   # (256, N) та (4, N)


# ─── Клас нейронної мережі ────────────────
class NeuralNet:
    def __init__(self, inp, l1, l2, out):
        np.random.seed(42)
        self.W1 = np.random.randn(l1, inp) * 0.1;  self.b1 = np.zeros((l1, 1))
        self.W2 = np.random.randn(l2, l1)  * 0.1;  self.b2 = np.zeros((l2, 1))
        self.W3 = np.random.randn(out, l2) * 0.1;  self.b3 = np.zeros((out, 1))

    def _parts(self):
        return [("W1",self.W1),("b1",self.b1),
                ("W2",self.W2),("b2",self.b2),
                ("W3",self.W3),("b3",self.b3)]

    def get_params(self):
        return np.concatenate([v.flatten() for _, v in self._parts()])

    def set_params(self, p):
        i = 0
        for name, arr in self._parts():
            s = arr.size
            setattr(self, name, p[i:i+s].reshape(arr.shape))
            i += s

    def forward(self, X):
        z1 = self.W1 @ X + self.b1;  a1 = sigmoid(z1)   # logsig
        z2 = self.W2 @ a1 + self.b2; a2 = np.tanh(z2)   # tansig
        z3 = self.W3 @ a2 + self.b3; a3 = sigmoid(z3)   # вихід
        return a1, a2, a3, z1, z2, z3

    def predict(self, X):
        return self.forward(X)[2]

    def loss_grad(self, params, X, T):
        self.set_params(params)
        a1, a2, a3, z1, z2, z3 = self.forward(X)
        N    = X.shape[1]
        loss = np.mean((a3 - T) ** 2)

        d3  = 2*(a3-T)/N * sigmoid_d(z3)
        dW3 = d3 @ a2.T;   db3 = d3.sum(1, keepdims=True)
        d2  = (self.W3.T @ d3) * tanh_d(z2)
        dW2 = d2 @ a1.T;   db2 = d2.sum(1, keepdims=True)
        d1  = (self.W2.T @ d2) * sigmoid_d(z1)
        dW1 = d1 @ X.T;    db1 = d1.sum(1, keepdims=True)

        grad = np.concatenate([g.flatten() for g in [dW1,db1,dW2,db2,dW3,db3]])
        return loss, grad


# ─── Навчання ─────────────────────────────
def train_network(net, P, T, mse_goal, max_epochs, max_time):
    history = []
    t0      = time.time()
    ep      = [0]

    def cb(params):
        ep[0] += 1
        loss, _ = net.loss_grad(params, P, T)
        history.append(loss)
        if ep[0] % 100 == 0:
            print(f"  Епоха {ep[0]:5d} | MSE = {loss:.8f} | {time.time()-t0:.1f}с")
        if loss <= mse_goal or time.time()-t0 >= max_time or ep[0] >= max_epochs:
            raise StopIteration

    print(f"\n▶  Навчання  (ціль MSE={mse_goal})")
    print("─" * 50)
    try:
        minimize(net.loss_grad, net.get_params(), args=(P, T), method="L-BFGS-B",
            jac=True, callback=cb,
            options={"maxiter": max_epochs, "ftol": 0, "gtol": 0})
    except StopIteration:
        pass

    elapsed = time.time() - t0
    print(f"\n✓  Готово | Епох: {ep[0]} | MSE: {history[-1]:.8f} | Час: {elapsed:.2f}с")
    return history, elapsed


# ─── Графік ───────────────────────────────
def plot_mse(history, mse_goal):
    plt.figure(figsize=(8, 4))
    plt.semilogy(history, color="royalblue", lw=2, label="Train MSE")
    plt.axhline(mse_goal, color="red", ls="--", label=f"Ціль = {mse_goal}")
    plt.xlabel("Епоха"); plt.ylabel("MSE"); plt.title("Помилка навчання")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    fname = f"mse_{mse_goal}.png"
    plt.savefig(fname, dpi=100); plt.show()
    print(f"  Графік: {fname}")


# ══════════════════════════════════════════
# ГОЛОВНА ПРОГРАМА
# ══════════════════════════════════════════
if __name__ == "__main__":
    # 1. Класи
    with open(CLASS_NAME_TY, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    num_classes = len(classes)
    print(f"Класи: {classes}")

    # 2. Зображення
    print("Завантаження зображень...")
    P, T = load_images(FILE_NAME_TX, TRAIN_FOLDER, num_classes)
    print(f"  P: {P.shape}  T: {T.shape}")

    # 3. Мережа
    net = NeuralNet(inp=P.shape[0], l1=LAYER1_NEURONS,
                    l2=LAYER2_NEURONS, out=num_classes)

    # 4. Навчання
    history, elapsed = train_network(net, P, T, MSE_GOAL, MAX_EPOCHS, MAX_TIME)

    # 5. Збереження
    save_name = f"net_{MSE_GOAL}.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(net, f)
    print(f"\n💾 Збережено: {save_name}")

    # 6. Графік
    plot_mse(history, MSE_GOAL)