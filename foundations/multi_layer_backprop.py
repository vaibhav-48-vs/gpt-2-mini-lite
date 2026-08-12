import numpy as np
from typing import List


class Solution:

    def ReLU(self, x):
        return np.maximum(0, x)

    def Linear(self, W, b, X):
        # X @ W.T + b
        return X @ W.T + b

    def getLoss(self, Ypred, Yorig):
        loss = np.mean((Ypred - Yorig) ** 2)
        return loss

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert lists to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)

        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)

        y_true = np.array(y_true, dtype=float)

        # =========================================================
        # FORWARD PASS
        # =========================================================

        # First linear layer
        z1 = self.Linear(W1, b1, x)

        # ReLU
        a1 = self.ReLU(z1)

        # Second linear layer
        z2 = self.Linear(W2, b2, a1)

        # Prediction
        y_pred = z2

        # MSE loss
        loss = self.getLoss(y_pred, y_true)

        # =========================================================
        # BACKWARD PASS
        # =========================================================

        # ---------------------------------------------------------
        # MSE derivative
        #
        # L = mean((y_pred - y_true)^2)
        #
        # dL/dy_pred = 2/n * (y_pred - y_true)
        # ---------------------------------------------------------

        n = len(y_true)

        dy_pred = (2 / n) * (y_pred - y_true)

        # Since y_pred = z2:
        #
        # dz2 = dy_pred
        dz2 = dy_pred

        # ---------------------------------------------------------
        # Second Linear Layer
        #
        # z2 = a1 @ W2.T + b2
        # ---------------------------------------------------------

        # dW2
        dW2 = np.outer(dz2, a1)

        # db2
        db2 = dz2.copy()

        # Gradient flowing back to a1
        da1 = dz2 @ W2

        # ---------------------------------------------------------
        # ReLU
        #
        # a1 = ReLU(z1)
        #
        # ReLU'(z1) = 1 if z1 > 0 else 0
        # ---------------------------------------------------------

        relu_mask = (z1 > 0).astype(float)

        dz1 = da1 * relu_mask

        # ---------------------------------------------------------
        # First Linear Layer
        #
        # z1 = x @ W1.T + b1
        # ---------------------------------------------------------

        # dW1
        dW1 = np.outer(dz1, x)

        # db1
        db1 = dz1.copy()

        # =========================================================
        # RETURN
        # =========================================================

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }