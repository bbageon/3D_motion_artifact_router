"""SequenceImitationSelector — RL-1 sequence-oracle imitation policy.

사용자 directive (2026-05-25):
> "RL-1 = sequence-oracle imitation policy (지도학습 sequence imitation, NOT
>  Q-learning, NOT 진짜 RL)."

State (16-dim):
  - current evaluator scores (3): max foot, bone, jitter.
  - score delta from previous step (3): t=0 모두 0.
  - step index (1).
  - previous tool one-hot (4): NONE / FootLock / BoneProjection / VelocitySmoothing.
  - previous strength one-hot (4): NONE / small / medium / large.
  - remaining budget (1) = K_max - step.

Action (10):
  0: STOP
  1-3: FootLock / (small, medium, large)
  4-6: BoneProjection / (small, medium, large)
  7-9: VelocitySmoothing / (small, medium, large)

Model: sklearn RandomForestClassifier 10-class joint.

NOTE: 본 selector 는 single-step prediction. closed-loop runner 가 매 step state 를
재계산해 selector.predict() 호출하면 sequence imitation 이 됨.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ALL_EVALUATORS = ("FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator")
TOOL_NAMES_WITH_NONE = ["NONE", "FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTH_NAMES_WITH_NONE = ["NONE", "small", "medium", "large"]
TOOL_TO_TARGET_PART = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}

ACTIONS: list[tuple[str, str]] = [
    ("STOP", "NONE"),
    ("FootLockTool", "small"), ("FootLockTool", "medium"), ("FootLockTool", "large"),
    ("BoneProjectionTool", "small"), ("BoneProjectionTool", "medium"), ("BoneProjectionTool", "large"),
    ("VelocitySmoothingTool", "small"), ("VelocitySmoothingTool", "medium"), ("VelocitySmoothingTool", "large"),
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
STATE_DIM = 16


@dataclass
class ImitationPrediction:
    action_id: int
    tool_name: str  # "STOP" / "FootLockTool" / ...
    strength: str   # "NONE" / "small" / "medium" / "large"
    target_part: str  # tool 의 natural target_part 또는 "n/a"
    is_stop: bool
    proba: Optional[list[float]] = None


def make_state(
    *,
    current_scores: dict[str, float],
    prev_scores: Optional[dict[str, float]],
    step_index: int,
    prev_tool: str,
    prev_strength: str,
    k_max: int,
) -> np.ndarray:
    """16-dim state vector."""
    score_vec = [current_scores[n] for n in ALL_EVALUATORS]
    if prev_scores is None:
        delta_vec = [0.0] * 3
    else:
        delta_vec = [current_scores[n] - prev_scores[n] for n in ALL_EVALUATORS]
    step_vec = [float(step_index)]
    tool_onehot = [1.0 if t == prev_tool else 0.0 for t in TOOL_NAMES_WITH_NONE]
    strength_onehot = [1.0 if s == prev_strength else 0.0 for s in STRENGTH_NAMES_WITH_NONE]
    budget_vec = [float(k_max - step_index)]
    vec = score_vec + delta_vec + step_vec + tool_onehot + strength_onehot + budget_vec
    arr = np.array(vec, dtype=np.float64)
    assert arr.shape == (STATE_DIM,), f"expected {STATE_DIM}-dim, got {arr.shape}"
    return arr


class SequenceImitationSelector:
    """sklearn classifier wrapper for 10-class action prediction."""

    def __init__(self, model_type: str = "random_forest", random_state: int = 1,
                 n_estimators: int = 100, **model_kwargs: Any) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model: Any = None
        self._is_trained = False
        self._trained_classes: list[int] = []

    def train(self, X: np.ndarray, y_action_id: np.ndarray) -> dict[str, Any]:
        """X: [N, 16]. y: [N] action_id ∈ [0, 9]."""
        if X.shape[1] != STATE_DIM:
            raise ValueError(f"X must be [N, {STATE_DIM}], got {X.shape}")
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.random_state,
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.random_state, max_iter=1000, multi_class="multinomial",
            )
        else:
            raise ValueError(self.model_type)
        self.model.fit(X, y_action_id)
        self._is_trained = True
        self._trained_classes = [int(c) for c in self.model.classes_]
        train_acc = float(self.model.score(X, y_action_id))
        return {
            "model_type": self.model_type,
            "n_train": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_accuracy": train_acc,
            "classes_seen": self._trained_classes,
        }

    def predict(self, state: np.ndarray, return_proba: bool = True) -> ImitationPrediction:
        if not self._is_trained or self.model is None:
            raise RuntimeError("Selector not trained.")
        if state.ndim == 1:
            state = state.reshape(1, -1)
        action_id = int(self.model.predict(state)[0])
        proba: Optional[list[float]] = None
        if return_proba and hasattr(self.model, "predict_proba"):
            # Expand proba to full 10-action space (untrained classes get 0).
            raw = self.model.predict_proba(state)[0]
            full = [0.0] * len(ACTIONS)
            for cls_idx, p in zip(self._trained_classes, raw):
                full[int(cls_idx)] = float(p)
            proba = full
        tool_name, strength = ACTIONS[action_id]
        is_stop = (tool_name == "STOP")
        target_part = TOOL_TO_TARGET_PART.get(tool_name, "n/a")
        return ImitationPrediction(
            action_id=action_id, tool_name=tool_name, strength=strength,
            target_part=target_part, is_stop=is_stop, proba=proba,
        )


def pairs_to_arrays(pairs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """training_data_v1.json 의 pairs 를 (X, y) 로 변환."""
    X = np.array([p["state"] for p in pairs], dtype=np.float64)
    y = np.array([p["action_id"] for p in pairs], dtype=np.int64)
    return X, y


def split_train_eval_by_trial(
    pairs: list[dict[str, Any]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample-level disjoint split — 같은 trial_id 가 train + eval 에 모두 안 들어감.

    distribution 별로 (synthetic_multi vs g2_natural) 같은 비율로 split.
    """
    rng = np.random.default_rng(seed)
    by_dist: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for p in pairs:
        d = p.get("distribution", "unknown")
        t = p["trial_id"]
        by_dist.setdefault(d, {}).setdefault(t, []).append(p)
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for dist_key, trials in by_dist.items():
        trial_ids = sorted(trials.keys())
        rng.shuffle(trial_ids)
        n_train = int(len(trial_ids) * train_ratio)
        train_ids = set(trial_ids[:n_train])
        for tid, ps in trials.items():
            (train if tid in train_ids else eval_).extend(ps)
    return train, eval_
