"""Supervised selector (B6 — Stage 0) — H-2026-205.

Oracle (B8) 의 best (tool, strength) label 을 사용해 (artifact_state) →
(tool, strength) 매핑을 supervised 학습.

학습 algorithm: scikit-learn 의 RandomForest / LogisticRegression / DummyClassifier.
single classifier 두 개:
  - tool_clf: state → tool_name (3-class).
  - strength_clf: state → strength (3-class).
또는 joint 9-class classifier.

State (input features):
  - artifact_kind one-hot (3).
  - evaluator scores (3: FootFloating / BoneLength / VelocityJitter max scores).
  → 6-dim feature vector.

Label:
  - tool_name (3-class: FootLockTool / BoneProjectionTool / VelocitySmoothingTool).
  - strength (3-class: small / medium / large).
  - target_part 는 학습 안 함 (tool 의 natural target_part 사용 또는 ARTIFACT_TOOL_CONFIG_OVERRIDE).

Train/eval split:
  - sample 단위 disjoint (같은 trial_id 가 train + eval 에 모두 안 들어감).
  - default: 70% train / 30% eval (random seed 고정).

본 module 은 train() + predict() interface 제공. measurement script
(`baseline_supervised_run.py`) 가 본 module 의 predict() 를 호출.

명세 §6.4.2 의 supervised selector 의 prototype. 본 prototype 의 NetGain 이 B5
(rule-based) 보다 의미 있게 개선되는지가 [H-2026-205](../evals/hypotheses/H-2026-205.md)
의 정량 검증.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

# scikit-learn 은 motion3d 환경에 이미 설치.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
except ImportError as e:
    raise ImportError(
        "scikit-learn required for SupervisedSelector. Install via pip install scikit-learn"
    ) from e

ARTIFACT_KINDS = ["foot_floating", "bone_stretch_right_arm", "global_jitter"]
TOOL_NAMES = ["FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTHS = ["small", "medium", "large"]
EVALUATOR_NAMES = ["FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator"]

#: artifact_kind → tool 의 natural target_part (B6 의 default).
ARTIFACT_TO_TARGET_PART: dict[str, str] = {
    "foot_floating": "both_feet",
    "bone_stretch_right_arm": "right_arm",
    "global_jitter": "full_body",
}

ModelType = Literal["random_forest", "logistic_regression", "dummy_most_frequent"]


@dataclass
class SupervisedSelectorPrediction:
    """B6 predict() 결과."""
    tool_name: str
    strength: str
    target_part: str
    tool_proba: dict[str, float] = field(default_factory=dict)
    strength_proba: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SupervisedSelector:
    """Supervised selector (B6 Stage 0).

    Two independent classifiers:
      - tool_clf: state → tool_name.
      - strength_clf: state → strength.
    target_part 은 artifact_kind 에서 결정 (ARTIFACT_TO_TARGET_PART).
    """

    def __init__(
        self,
        model_type: ModelType = "random_forest",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.tool_clf: Optional[Any] = None
        self.strength_clf: Optional[Any] = None
        self._feature_names: Optional[list[str]] = None

    def _make_classifier(self) -> Any:
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state)
        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=500, random_state=self.random_state)
        if self.model_type == "dummy_most_frequent":
            return DummyClassifier(strategy="most_frequent", random_state=self.random_state)
        raise ValueError(f"unknown model_type {self.model_type!r}")

    def train(
        self,
        X: np.ndarray,
        tool_labels: np.ndarray,
        strength_labels: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Train two classifiers.

        Args:
            X: [n_samples, n_features] feature matrix.
            tool_labels: [n_samples] tool_name list (string).
            strength_labels: [n_samples] strength list (string).
            feature_names: optional list of feature names (for inspection).

        Returns:
            {"tool_train_score", "strength_train_score", "model_type"}.
        """
        self._feature_names = feature_names
        self.tool_clf = self._make_classifier()
        self.strength_clf = self._make_classifier()
        self.tool_clf.fit(X, tool_labels)
        self.strength_clf.fit(X, strength_labels)
        return {
            "model_type": self.model_type,
            "n_train": X.shape[0],
            "tool_train_accuracy": float(self.tool_clf.score(X, tool_labels)),
            "strength_train_accuracy": float(self.strength_clf.score(X, strength_labels)),
        }

    def predict(self, state_vector: np.ndarray, artifact_kind: str) -> SupervisedSelectorPrediction:
        """state vector + artifact_kind → SupervisedSelectorPrediction.

        Args:
            state_vector: [n_features] single sample feature vector.
            artifact_kind: 'foot_floating' / 'bone_stretch_right_arm' / 'global_jitter'.
        """
        if self.tool_clf is None or self.strength_clf is None:
            raise RuntimeError("SupervisedSelector not trained yet — call train() first.")
        X = state_vector.reshape(1, -1)
        tool_pred = str(self.tool_clf.predict(X)[0])
        strength_pred = str(self.strength_clf.predict(X)[0])

        # probabilities (if classifier supports).
        tool_proba: dict[str, float] = {}
        strength_proba: dict[str, float] = {}
        if hasattr(self.tool_clf, "predict_proba"):
            probs = self.tool_clf.predict_proba(X)[0]
            classes = list(self.tool_clf.classes_)
            tool_proba = {str(c): float(p) for c, p in zip(classes, probs)}
        if hasattr(self.strength_clf, "predict_proba"):
            probs = self.strength_clf.predict_proba(X)[0]
            classes = list(self.strength_clf.classes_)
            strength_proba = {str(c): float(p) for c, p in zip(classes, probs)}

        target_part = ARTIFACT_TO_TARGET_PART.get(artifact_kind, "full_body")
        return SupervisedSelectorPrediction(
            tool_name=tool_pred,
            strength=strength_pred,
            target_part=target_part,
            tool_proba=tool_proba,
            strength_proba=strength_proba,
            metadata={
                "model_type": self.model_type,
                "artifact_kind": artifact_kind,
            },
        )

    def evaluate(
        self,
        X_eval: np.ndarray,
        tool_eval_labels: np.ndarray,
        strength_eval_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Compute eval-set accuracy / agreement."""
        if self.tool_clf is None or self.strength_clf is None:
            raise RuntimeError("SupervisedSelector not trained yet — call train() first.")
        tool_pred = self.tool_clf.predict(X_eval)
        strength_pred = self.strength_clf.predict(X_eval)
        tool_acc = float(np.mean(tool_pred == tool_eval_labels))
        strength_acc = float(np.mean(strength_pred == strength_eval_labels))
        joint_acc = float(np.mean((tool_pred == tool_eval_labels) & (strength_pred == strength_eval_labels)))
        return {
            "model_type": self.model_type,
            "n_eval": X_eval.shape[0],
            "tool_accuracy": tool_acc,
            "strength_accuracy": strength_acc,
            "joint_accuracy": joint_acc,
        }


def split_train_eval_by_trial(
    tuples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """sample (trial_id) 단위로 disjoint split.

    AGENTS.md §3-15 의 split_id 의무 — 본 split 은 'supervised_selector_v1' 으로
    raw record 에 박제.
    """
    rng = np.random.default_rng(seed)
    trial_ids = sorted(set(t["trial_id"] for t in tuples))
    rng.shuffle(trial_ids)
    n_train = int(len(trial_ids) * train_ratio)
    train_trial_ids = set(trial_ids[:n_train])
    eval_trial_ids = set(trial_ids[n_train:])
    train_tuples = [t for t in tuples if t["trial_id"] in train_trial_ids]
    eval_tuples = [t for t in tuples if t["trial_id"] in eval_trial_ids]
    return train_tuples, eval_tuples


def tuples_to_arrays(tuples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tuples → (X, tool_labels, strength_labels)."""
    X = np.array([t["state"]["feature_vector"] for t in tuples], dtype=np.float64)
    tool_labels = np.array([t["label"]["best_tool"] for t in tuples])
    strength_labels = np.array([t["label"]["best_strength"] for t in tuples])
    return X, tool_labels, strength_labels
