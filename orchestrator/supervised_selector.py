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

#: artifact_kind → tool name (rule-based mapping, used by
#: ablation_mode='rule_tool_learned_strength').
ARTIFACT_TO_TARGET_TOOL: dict[str, str] = {
    "foot_floating": "FootLockTool",
    "bone_stretch_right_arm": "BoneProjectionTool",
    "global_jitter": "VelocitySmoothingTool",
}

ModelType = Literal["random_forest", "logistic_regression", "dummy_most_frequent"]
AblationMode = Literal["none", "score_only", "rule_tool_learned_strength"]


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
        ablation_mode: AblationMode = "none",
    ) -> None:
        """Args:
            ablation_mode:
              - 'none' (default): full features (artifact_kind onehot + evaluator scores) + tool/strength 둘 다 학습.
              - 'score_only': artifact_kind onehot 제거, evaluator scores 만 사용 → 일반화 가능성 검증.
              - 'rule_tool_learned_strength': tool 은 ARTIFACT_TO_TARGET_TOOL rule lookup,
                strength 만 학습 → 학습 신호의 strength-isolation.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.ablation_mode = ablation_mode
        self.tool_clf: Optional[Any] = None
        self.strength_clf: Optional[Any] = None
        self._feature_names: Optional[list[str]] = None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """ablation_mode 에 따라 feature 변환.

        Default state vector schema: [artifact_onehot(3), evaluator_scores(3)] = 6-dim.
        - 'score_only': first 3 (artifact_onehot) 제거 → 3-dim.
        - others: pass-through.
        """
        if self.ablation_mode == "score_only":
            return X[:, 3:] if X.ndim == 2 else X[3:].reshape(1, -1)
        return X

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
        X_proc = self._preprocess(X)
        # tool_clf: ablation_mode='rule_tool_learned_strength' 이면 학습 안 함.
        if self.ablation_mode == "rule_tool_learned_strength":
            self.tool_clf = None
            tool_train_acc = float("nan")
        else:
            self.tool_clf = self._make_classifier()
            self.tool_clf.fit(X_proc, tool_labels)
            tool_train_acc = float(self.tool_clf.score(X_proc, tool_labels))
        # strength_clf: 항상 학습.
        self.strength_clf = self._make_classifier()
        self.strength_clf.fit(X_proc, strength_labels)
        return {
            "model_type": self.model_type,
            "ablation_mode": self.ablation_mode,
            "n_train": X.shape[0],
            "n_features_after_preprocess": X_proc.shape[1] if X_proc.ndim == 2 else len(X_proc),
            "tool_train_accuracy": tool_train_acc,
            "strength_train_accuracy": float(self.strength_clf.score(X_proc, strength_labels)),
        }

    def predict(self, state_vector: np.ndarray, artifact_kind: str) -> SupervisedSelectorPrediction:
        """state vector + artifact_kind → SupervisedSelectorPrediction."""
        if self.strength_clf is None:
            raise RuntimeError("SupervisedSelector not trained yet — call train() first.")
        X = state_vector.reshape(1, -1)
        X_proc = self._preprocess(X)

        # Tool prediction: rule or learned.
        tool_proba: dict[str, float] = {}
        if self.ablation_mode == "rule_tool_learned_strength":
            tool_pred = ARTIFACT_TO_TARGET_TOOL.get(artifact_kind, "FootLockTool")
            tool_proba = {tool_pred: 1.0}
        else:
            if self.tool_clf is None:
                raise RuntimeError("tool_clf 가 None — ablation_mode 와 train state 불일치.")
            tool_pred = str(self.tool_clf.predict(X_proc)[0])
            if hasattr(self.tool_clf, "predict_proba"):
                probs = self.tool_clf.predict_proba(X_proc)[0]
                classes = list(self.tool_clf.classes_)
                tool_proba = {str(c): float(p) for c, p in zip(classes, probs)}

        # Strength prediction: 항상 learned.
        strength_pred = str(self.strength_clf.predict(X_proc)[0])
        strength_proba: dict[str, float] = {}
        if hasattr(self.strength_clf, "predict_proba"):
            probs = self.strength_clf.predict_proba(X_proc)[0]
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
                "ablation_mode": self.ablation_mode,
                "artifact_kind": artifact_kind,
            },
        )

    def evaluate(
        self,
        X_eval: np.ndarray,
        tool_eval_labels: np.ndarray,
        strength_eval_labels: np.ndarray,
        artifact_kinds: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Compute eval-set accuracy / agreement.

        Args:
            artifact_kinds: ablation_mode='rule_tool_learned_strength' 시 tool prediction
                위해 필요. None 이면 tool_acc 계산 안 함 (NaN).
        """
        if self.strength_clf is None:
            raise RuntimeError("SupervisedSelector not trained yet — call train() first.")
        X_proc = self._preprocess(X_eval)
        # Tool predictions.
        if self.ablation_mode == "rule_tool_learned_strength":
            if artifact_kinds is None:
                tool_pred = np.array(["UNKNOWN"] * X_eval.shape[0])
            else:
                tool_pred = np.array([ARTIFACT_TO_TARGET_TOOL.get(a, "UNKNOWN") for a in artifact_kinds])
        else:
            if self.tool_clf is None:
                raise RuntimeError("tool_clf is None — ablation_mode mismatch.")
            tool_pred = self.tool_clf.predict(X_proc)
        strength_pred = self.strength_clf.predict(X_proc)
        tool_acc = float(np.mean(tool_pred == tool_eval_labels))
        strength_acc = float(np.mean(strength_pred == strength_eval_labels))
        joint_acc = float(np.mean((tool_pred == tool_eval_labels) & (strength_pred == strength_eval_labels)))
        return {
            "model_type": self.model_type,
            "ablation_mode": self.ablation_mode,
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
