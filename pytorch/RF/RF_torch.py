import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

def _bootstrap_indices(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_samples, size=n_samples, dtype=np.int64)

def _choose_features(n_features: int, max_features, rng: np.random.Generator) -> np.ndarray:
    # max_features: 'sqrt' | 'log2' | int | float(0-1)
    if isinstance(max_features, str):
        if max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError("Unknown max_features")
    elif isinstance(max_features, float):
        k = max(1, int(round(n_features * max(0.0, min(1.0, max_features)))))
    elif isinstance(max_features, int):
        k = max(1, min(n_features, max_features))
    elif max_features is None:
        k = n_features
    else:
        raise ValueError("Bad max_features")
    return np.sort(rng.choice(n_features, size=k, replace=False).astype(np.int64))

class _CARTTreeBuilder:
    def __init__(
        self,
        task: str,  # 'classifier' | 'regressor'
        n_classes: Optional[int] = None,
        max_depth: int = 16,
        min_samples_split: int = 2,
        max_features: Union[str, int, float, None] = "sqrt",
        random_state: Optional[int] = None,
    ):
        assert task in ("classifier", "regressor")
        self.task = task
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

        # 将要导出的结构
        self.children_left: List[int] = []
        self.children_right: List[int] = []
        self.feature: List[int] = []
        self.threshold: List[float] = []
        # 叶子值：回归：(1,)；分类：(n_classes,)
        self.value: List[np.ndarray] = []

    # impurity 与 leaf value
    def _gini(self, y: np.ndarray) -> float:
        # y: (n,)
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        p = counts / counts.sum()
        return 1.0 - np.sum(p * p)

    def _mse(self, y: np.ndarray) -> float:
        mu = y.mean()
        return np.mean((y - mu) ** 2)

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        if self.task == "regressor":
            return np.array([y.mean()], dtype=np.float32)
        else:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
            return counts  # 保存原始计数，推理时再转概率

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self.task == "classifier" else self._mse(y)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        返回 (best_feat, best_thr, best_gain)
        如果找不到有效划分，返回 (-1, np.nan, 0.0)
        """
        n, _ = X.shape
        parent_imp = self._impurity(y)
        best_gain = 0.0
        best_feat = -1
        best_thr = np.nan

        for f in feat_idx:
            x = X[:, f]
            order = np.argsort(x, kind="mergesort")
            x_sorted = x[order]
            y_sorted = y[order]

            # 为了效率，考虑相邻唯一阈值
            unique_mask = x_sorted[1:] != x_sorted[:-1]
            if not np.any(unique_mask):
                continue

            # 前缀计数/均值以快速评估
            if self.task == "classifier":
                cum_counts = np.zeros((n, self.n_classes), dtype=np.int64)
                for i in range(n):
                    cum_counts[i, y_sorted[i]] += 1
                    if i > 0:
                        cum_counts[i] += cum_counts[i - 1]
                total_counts = cum_counts[-1].astype(np.float64)
                for i in np.where(unique_mask)[0]:
                    n_left = i + 1
                    n_right = n - n_left
                    if n_left < self.min_samples_split or n_right < self.min_samples_split:
                        continue
                    left_counts = cum_counts[i].astype(np.float64)
                    right_counts = total_counts - left_counts
                    pL = left_counts / left_counts.sum()
                    pR = right_counts / right_counts.sum()
                    giniL = 1.0 - np.sum(pL * pL)
                    giniR = 1.0 - np.sum(pR * pR)
                    gain = parent_imp - (n_left / n) * giniL - (n_right / n) * giniR
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = int(f)
                        best_thr = (x_sorted[i] + x_sorted[i + 1]) * 0.5
            else:
                # 回归：用累积和计算 MSE
                cumsum = np.cumsum(y_sorted, dtype=np.float64)
                cumsum2 = np.cumsum(y_sorted.astype(np.float64) ** 2)
                sumY = cumsum[-1]
                sumY2 = cumsum2[-1]
                for i in np.where(unique_mask)[0]:
                    n_left = i + 1
                    n_right = n - n_left
                    if n_left < self.min_samples_split or n_right < self.min_samples_split:
                        continue
                    sumL = cumsum[i]
                    sumL2 = cumsum2[i]
                    sumR = sumY - sumL
                    sumR2 = sumY2 - sumL2
                    mseL = (sumL2 / n_left) - (sumL / n_left) ** 2
                    mseR = (sumR2 / n_right) - (sumR / n_right) ** 2
                    gain = parent_imp - (n_left / n) * mseL - (n_right / n) * mseR
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = int(f)
                        best_thr = (x_sorted[i] + x_sorted[i + 1]) * 0.5

        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> int:
        """
        递归建树。返回当前节点 index。
        我们把树存为并行数组，新增节点时，把占位写入，再回填左右孩子索引。
        """
        node_id = len(self.feature)
        # 先占位
        self.feature.append(-2)    # 叶子用 -2
        self.threshold.append(np.nan)
        self.children_left.append(-1)
        self.children_right.append(-1)
        self.value.append(self._leaf_value(y))

        # 停止条件
        if depth >= self.max_depth or y.size < self.min_samples_split:
            return node_id
        if self.task == "classifier" and np.unique(y).size == 1:
            return node_id

        feat_idx = _choose_features(X.shape[1], self.max_features, self.rng)
        best_feat, best_thr, best_gain = self._best_split(X, y, feat_idx)

        if best_feat == -1 or best_gain <= 0.0:
            return node_id

        # 真正分裂
        mask_left = X[:, best_feat] <= best_thr
        X_L, y_L = X[mask_left], y[mask_left]
        X_R, y_R = X[~mask_left], y[~mask_left]
        if X_L.size == 0 or X_R.size == 0:
            return node_id

        # 更新当前节点信息（非叶）
        self.feature[node_id] = best_feat
        self.threshold[node_id] = float(best_thr)

        # 左右子树
        left_id = self._build(X_L, y_L, depth + 1)
        right_id = self._build(X_R, y_R, depth + 1)
        self.children_left[node_id] = left_id
        self.children_right[node_id] = right_id
        return node_id

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.children_left.clear(); self.children_right.clear()
        self.feature.clear(); self.threshold.clear(); self.value.clear()
        self._build(X, y, depth=0)

    def export(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 导出为 numpy 数组
        children_left = np.asarray(self.children_left, dtype=np.int64)
        children_right = np.asarray(self.children_right, dtype=np.int64)
        feature = np.asarray(self.feature, dtype=np.int64)
        threshold = np.asarray(self.threshold, dtype=np.float32)
        # value: list of arrays -> stack
        v0 = self.value[0]
        if v0.ndim == 1:
            n_nodes = len(self.value)
            n_out = v0.shape[0]
            value = np.zeros((n_nodes, n_out), dtype=np.float32)
            for i, arr in enumerate(self.value):
                value[i, :arr.shape[0]] = arr
        else:
            raise RuntimeError("Unexpected value shape")
        return children_left, children_right, feature, threshold, value


class _DecisionTreeTS(nn.Module):
    """通用树：分类/回归都能用。分类 value 为计数向量，回归 value 为形状(1,)"""

    def __init__(
        self,
        children_left: torch.Tensor,   # (n_nodes,) int64
        children_right: torch.Tensor,  # (n_nodes,) int64
        feature: torch.Tensor,         # (n_nodes,) int64  叶子为 -2
        threshold: torch.Tensor,       # (n_nodes,) float32
        value: torch.Tensor,           # (n_nodes, n_out) float32; 分类n_out=n_classes，回归n_out=1
        task: str,                     # 'classifier' | 'regressor'
    ):
        super().__init__()
        self.register_buffer("children_left", children_left.to(torch.long))
        self.register_buffer("children_right", children_right.to(torch.long))
        self.register_buffer("feature", feature.to(torch.long))
        self.register_buffer("threshold", threshold.to(torch.float32))
        self.register_buffer("value", value.to(torch.float32))
        self.task = task
        self.n_out = value.size(1)

    @torch.jit.export
    def predict_one(self, x: torch.Tensor) -> torch.Tensor:
        # 显式类型，避免 TorchScript 推成 "number"
        node: int = 0
        max_steps: int = int(self.feature.numel())  # 访问节点数的安全上界

        step: int = 0
        while step < max_steps:
            cl: int = int(self.children_left[node].item())
            cr: int = int(self.children_right[node].item())

            # 叶子：直接返回
            if cl == -1 and cr == -1:
                out = self.value[node]  # (n_out,)
                if self.task == "classifier":
                    s_t = torch.sum(out)                # 标量 tensor
                    s: float = float(s_t.item())        # Python float 用于分支判断
                    if s > 0.0:
                        return out / s_t                # 保持张量运算
                    else:
                        return torch.full_like(out, 1.0 / float(self.n_out))
                else:
                    return out  # 回归：(1,)

            f: int = int(self.feature[node].item())
            if f == -2:
                # 非法/兜底：按当前节点值返回，确保有返回
                return self.value[node]

            thr: float = float(self.threshold[node].item())
            x_f: float = float(x[f].item())  # 索引用 int，比较用 float

            if x_f <= thr:
                node = cl
            else:
                node = cr

            step += 1

        # 兜底（理论上不会走到）：返回当前节点值，保证所有路径都有返回
        return self.value[node]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N: int = int(X.size(0))
        out = torch.empty((N, self.n_out), dtype=torch.float32, device=X.device)
        i: int = 0
        while i < N:
            out[i] = self.predict_one(X[i])
            i += 1
        return out


class RandomForestTS(nn.Module):
    """支持分类/回归；分类输出概率，回归输出 (N,1)"""

    def __init__(self, trees: List[_DecisionTreeTS], task: str):
        super().__init__()
        self.trees = nn.ModuleList(trees)
        self.task = task
        self.n_out = trees[0].n_out if len(trees) > 0 else 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros(X.size(0), self.n_out, dtype=torch.float32, device=X.device)
        for t in self.trees:
            agg += t(X)
        agg /= max(1, len(self.trees))
        return agg

    @torch.jit.export
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.task == "classifier":
            probs = self.forward(X)
            return torch.argmax(probs, dim=1)
        else:
            return self.forward(X).squeeze(-1)  # (N,)


class RandomForest:
    """
    训练器（numpy 实现），fit 完后用 as_torch_module() 得到可 TorchScript 的 nn.Module。
    """

    def __init__(
        self,
        task: str,  # 'classifier' | 'regressor'
        n_estimators: int = 100,
        max_depth: int = 16,
        min_samples_split: int = 2,
        max_features: Union[str, int, float, None] = "sqrt",
        random_state: Optional[int] = None,
    ):
        assert task in ("classifier", "regressor")
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.builders: List[_CARTTreeBuilder] = []
        self.n_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if self.task == "classifier":
            y = np.asarray(y, dtype=np.int64)
            self.n_classes = int(np.max(y)) + 1
        else:
            y = np.asarray(y, dtype=np.float64)

        n_samples = X.shape[0]
        base_rng = np.random.default_rng(self.random_state)
        self.builders.clear()

        for m in range(self.n_estimators):
            idx = _bootstrap_indices(n_samples, base_rng)
            builder = _CARTTreeBuilder(
                task=self.task,
                n_classes=self.n_classes,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(base_rng.integers(0, 2**31 - 1)),
            )
            builder.fit(X[idx], y[idx])
            self.builders.append(builder)

    def as_torch_module(self, device: Optional[torch.device] = None) -> RandomForestTS:
        trees: List[_DecisionTreeTS] = []
        for b in self.builders:
            cl, cr, feat, thr, val = b.export()
            t = _DecisionTreeTS(
                torch.from_numpy(cl),
                torch.from_numpy(cr),
                torch.from_numpy(feat),
                torch.from_numpy(thr),
                torch.from_numpy(val),
                task=self.task,
            )
            if device is not None:
                t.to(device)
            trees.append(t)
        return RandomForestTS(trees, task=self.task)

if __name__ == "__main__":
    # Classify
    rng = np.random.default_rng(0)
    # Xc = rng.normal(size=(400, 6))
    # yc = (Xc[:, 0] + Xc[:, 1] * 0.5 > 0).astype(np.int64)

    # clf = RandomForest(task="classifier", n_estimators=20, max_depth=10, random_state=0)
    # clf.fit(Xc, yc)

    # clf_torch = clf.as_torch_module()
    # clf_torch.eval()

    # Xc_t = torch.from_numpy(Xc[:5]).to(torch.float32)
    # probs = clf_torch(Xc_t)              # (5, n_classes)
    # pred = clf_torch.predict(Xc_t)       # (5,)
    # print("clf probs:", probs, "\nclf pred:", pred)

    # # TorchScript
    # scripted_clf = torch.jit.script(clf_torch)
    # scripted_clf.save("rf_classifier_ts.pt")

    # Regress
    Xr = rng.normal(size=(400, 10))  # 400 样本，10 个特征
    print(Xr[0,:])
    yr = 3.0 * Xr[:, 0] - 2.0 * Xr[:, 1] + rng.normal(scale=0.3, size=400)

    rgr = RandomForest(task="regressor", n_estimators=30, max_depth=12, random_state=0)
    rgr.fit(Xr, yr)

    rgr_torch = rgr.as_torch_module()
    rgr_torch.eval()

    Xr_t = torch.from_numpy(Xr[:5]).to(torch.float32)
    print(Xr_t[0,:])
    yhat = rgr_torch(Xr_t)               # (5, 1)
    print("rgr pred:", yhat.squeeze(-1))

    scripted_rgr = torch.jit.script(rgr_torch)
    scripted_rgr.save("rf_regressor_ts.pt")

