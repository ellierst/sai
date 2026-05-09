"""
Лабораторна робота №3 — Варіант 11
Розпізнавання зображень за допомогою навченої нейромережі
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# ══════════════════════════════════════════
# НАЛАШТУВАННЯ
# ══════════════════════════════════════════
RECOGN_FOLDER = "Neuro_Recogn"
CLASS_NAME_TY = "Neuro_Train/Class_Name_ty1.txt"
NET_FILE      = "net_0.0001.pkl"   # змінюйте: net_0.01.pkl / net_0.001.pkl / net_0.0001.pkl
# ══════════════════════════════════════════


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_d(x):
    return 1.0 - np.tanh(x) ** 2

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
        z1 = self.W1 @ X + self.b1;  a1 = sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2; a2 = np.tanh(z2)
        z3 = self.W3 @ a2 + self.b3; a3 = sigmoid(z3)
        return a1, a2, a3, z1, z2, z3

    def predict(self, X):
        return self.forward(X)[2]


def load_net(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def recognize_image(net, img_path):
    img = Image.open(img_path).convert("L").resize((16, 16))
    vec = np.array(img, dtype=np.float64).flatten() / 255.0
    X   = vec.reshape(-1, 1)
    Y   = net.predict(X).flatten()
    return Y


if __name__ == "__main__":
    # Завантажити класи
    with open(CLASS_NAME_TY, "r", encoding="utf-8") as f:
        classes = [l.strip().split(None, 1)[1] for l in f if l.strip()]
    print(f"Класи: {classes}\n")

    # Завантажити мережу
    net = load_net(NET_FILE)
    print(f"Мережу завантажено: {NET_FILE}\n")

    # Знайти всі зображення
    images = sorted([
        f for f in os.listdir(RECOGN_FOLDER)
        if f.lower().endswith(".bmp")
    ])

    print(f"{'Зображення':<25} {'Вектор Y':<45} {'Результат'}")
    print("─" * 85)

    results = []
    for img_name in images:
        path = os.path.join(RECOGN_FOLDER, img_name)
        Y    = recognize_image(net, path)

        predicted_idx   = np.argmax(Y)
        predicted_class = classes[predicted_idx]
        y_str = "[" + ", ".join(f"{v:.3f}" for v in Y) + "]"

        print(f"{img_name:<25} {y_str:<45} → {predicted_class}")
        results.append((img_name, Y, predicted_class))

    # ─── Діаграми ───────────────────────────────
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()

    for i, (name, Y, pred) in enumerate(results):
        ax = axes[i]
        bars = ax.bar(classes, Y, color=["royalblue"]*len(classes))
        bars[np.argmax(Y)].set_color("tomato")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name}\n→ {pred}", fontsize=8)
        ax.set_ylabel("Y", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    # Сховати порожні
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Розпізнавання ({NET_FILE})", fontsize=11, y=1.01)
    plt.tight_layout()
    out_name = f"recognition_{NET_FILE.replace('.pkl','')}.png"
    plt.savefig(out_name, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"\nДіаграми збережено: {out_name}")