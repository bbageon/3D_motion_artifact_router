"""Contextual bandit (offline Q-regression) — H-2026-205 Step 4.

State (3-dim, full evaluator scores including BoneLength):
  - FootFloatingEvaluator max score
  - BoneLengthEvaluator max score   ← Step 3.5 와 달리 포함! (사용자 directive)
  - VelocityJitterEvaluator max score

Action (10 classes):
  0 = STOP (do nothing)
  1-9 = FootLockTool/{small,medium,large}, BoneProjectionTool/..., VelocitySmoothingTool/...

Algorithm:
  - Offline Q-regression. (state, action_onehot) → reward.
  - RandomForestRegressor or LinearRegression.
  - Greedy: argmax_a Q(state, a).

본 module 은 H-2026-205 의 사전 등록 contextual bandit (LinUCB / Thompson sampling)
의 **offline RandomForest variant**. exploration 없음 — 학습 data 의 모든 action 의
reward 가 사전 측정되어 있음. LinUCB 의 exploration term 의 의미는 online learning
시점에 발생, offline simulator data 에서는 무관.

Multi-step value learning (cumulative reward over trajectories) 은 후속 Stage 1
(H-2026-205 Stage 1) 또는 Stage 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    raise ImportError("scikit-learn required") from e

STATE_FEATURES = ["FootFloatingEvaluator", "BoneLengthEvaluator", "VelocityJitterEvaluator"]
TOOL_NAMES = ["FootLockTool", "BoneProjectionTool", "VelocitySmoothingTool"]
STRENGTHS = ["small", "medium", "large"]

#: action_id → (tool, strength). 0 = STOP, 1..9 = 3×3.
ACTIONS: list[tuple[str, str]] = [("STOP", "none")]
for tn in TOOL_NAMES:
    for st in STRENGTHS:
        ACTIONS.append((tn, st))

#: action_id → (tool, strength) dict.
ACTION_MAP: dict[int, tuple[str, str]] = {i: a for i, a in enumerate(ACTIONS)}

#: artifact_kind → tool 의 natural target_part.
ARTIFACT_TO_TARGET_PART: dict[str, str] = {
    "foot_floating": "both_feet",
    "bone_stretch_right_arm": "right_arm",
    "global_jitter": "full_body",
}
TOOL_TO_TARGET_PART: dict[str, str] = {
    "FootLockTool": "both_feet",
    "BoneProjectionTool": "right_arm",
    "VelocitySmoothingTool": "full_body",
}

ModelType = Literal["random_forest", "linear"]


@dataclass
class BanditPrediction:
    """Bandit predict() 결과."""
    action_id: int
    tool_name: str
    strength: str
    target_part: str
    q_values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextualBandit:
    """Offline Q-regression contextual bandit.

    Input: (state, action_onehot) features → Q (reward).
    Predict: argmax_a Q(state, a) for all 10 actions.
    """

    def __init__(
        self,
        model_type: ModelType = "random_forest",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.q_model: Optional[Any] = None
        self.n_actions: int = len(ACTIONS)
        self.state_dim: int = len(STATE_FEATURES)

    def _make_model(self) -> Any:
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=self.random_state)
        if self.model_type == "linear":
            return LinearRegression()
        raise ValueError(f"unknown model_type {self.model_type!r}")

    def _build_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """[N, state_dim] + [N] action_id → [N, state_dim + n_actions] (state + action one-hot)."""
        n = states.shape[0]
        action_onehot = np.zeros((n, self.n_actions), dtype=np.float64)
        action_onehot[np.arange(n), actions] = 1.0
        return np.hstack([states, action_onehot])

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> dict[str, Any]:
        """Train Q model. (state, action) → reward."""
        X = self._build_features(states, actions)
        self.q_model = self._make_model()
        self.q_model.fit(X, rewards)
        return {
            "model_type": self.model_type,
            "n_train": X.shape[0],
            "train_r2": float(self.q_model.score(X, rewards)),
        }

    def predict_q_all(self, state_vector: np.ndarray) -> np.ndarray:
        """Single state → Q values for all actions [n_actions]."""
        if self.q_model is None:
            raise RuntimeError("ContextualBandit not trained — call train() first.")
        # Build all (state, action) features.
        states_repeated = np.tile(state_vector.reshape(1, -1), (self.n_actions, 1))
        actions_all = np.arange(self.n_actions)
        X = self._build_features(states_repeated, actions_all)
        q_values = self.q_model.predict(X)
        return q_values

    def predict(self, state_vector: np.ndarray) -> BanditPrediction:
        """Greedy: argmax_a Q(state, a)."""
        q = self.predict_q_all(state_vector)
        best_idx = int(np.argmax(q))
        tool_name, strength = ACTIONS[best_idx]
        target_part = TOOL_TO_TARGET_PART.get(tool_name, "full_body") if tool_name != "STOP" else "n/a"
        return BanditPrediction(
            action_id=best_idx,
            tool_name=tool_name,
            strength=strength,
            target_part=target_part,
            q_values=q.tolist(),
            metadata={"model_type": self.model_type},
        )


class ContextualBanditOrchestrator:
    """ContextualBandit 의 Orchestrator interface wrapper.

    RefinementLoop 에 직접 plug-in 가능. evaluator reports → state → bandit predict
    → OrchestratorDecision.

    STOP 처리: action_id=0 (STOP) 가 max Q 면 decision=STOP.
    """

    name = "ContextualBanditOrchestrator"

    EVALUATOR_TO_FEATURE_IDX: dict[str, int] = {
        "FootFloatingEvaluator": 0,
        "BoneLengthEvaluator": 1,
        "VelocityJitterEvaluator": 2,
    }

    def __init__(
        self,
        bandit: ContextualBandit,
    ) -> None:
        self.bandit = bandit

    def decide(
        self,
        evaluator_reports: list[Any],
        tool_history: list[Any],
        **kwargs: Any,
    ) -> Any:
        from orchestrator.base import OrchestratorDecision

        # Group reports by agent.
        reports_by_agent: dict[str, list[Any]] = {}
        for r in evaluator_reports:
            reports_by_agent.setdefault(r.agent, []).append(r)

        # State vector: full evaluator max scores.
        state = np.array([
            max((r.score for r in reports_by_agent.get(name, [])), default=0.0)
            for name in STATE_FEATURES
        ], dtype=np.float64)

        # Bandit predict.
        pred = self.bandit.predict(state)

        if pred.action_id == 0:  # STOP
            return OrchestratorDecision(
                decision="STOP",
                next_step="STOP",
                metadata={
                    "orchestrator": self.name,
                    "stop_reason": "bandit_chose_stop",
                    "state": state.tolist(),
                    "q_values": pred.q_values,
                    "tool_history_len": len(tool_history),
                },
            )

        # Find primary_error for revise.
        # tool → primary evaluator (the one that this tool primarily addresses).
        tool_to_primary_eval = {
            "FootLockTool": "FootFloatingEvaluator",
            "BoneProjectionTool": "BoneLengthEvaluator",
            "VelocitySmoothingTool": "VelocityJitterEvaluator",
        }
        primary_eval = tool_to_primary_eval.get(pred.tool_name, "")
        primary_reports = reports_by_agent.get(primary_eval, [])
        primary_error = None
        target_frames = None
        if primary_reports:
            primary_reports.sort(key=lambda r: r.score, reverse=True)
            top = primary_reports[0]
            primary_error = top.error_type
            target_frames = tuple(top.frames) if top.frames else None

        return OrchestratorDecision(
            decision="revise",
            primary_error=primary_error,
            selected_tool=pred.tool_name,
            target_part=pred.target_part,
            target_frames=target_frames,
            strength=pred.strength,
            next_step="apply_then_evaluate",
            score=float(pred.q_values[pred.action_id]),
            metadata={
                "orchestrator": self.name,
                "action_id": pred.action_id,
                "q_values": pred.q_values,
                "state": state.tolist(),
                "tool_history_len": len(tool_history),
            },
        )

    def orchestrator_class_hash(self) -> str:
        import hashlib
        import inspect
        source = inspect.getsource(type(self))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()


def tuples_to_arrays(tuples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """training tuples → (states, actions, rewards). Skip 된 tuple 제거."""
    valid = [t for t in tuples if not t.get("skipped")]
    states = np.array([t["state"] for t in valid], dtype=np.float64)
    actions = np.array([t["action_id"] for t in valid], dtype=np.int64)
    rewards = np.array([t["reward_netgain"] for t in valid], dtype=np.float64)
    return states, actions, rewards


def split_train_eval_by_trial(
    tuples: list[dict[str, Any]],
    train_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """sample (trial_id) 단위 disjoint split."""
    rng = np.random.default_rng(seed)
    trial_ids = sorted({t["trial_id"] for t in tuples})
    rng.shuffle(trial_ids)
    n_train = int(len(trial_ids) * train_ratio)
    train_set = set(trial_ids[:n_train])
    eval_set = set(trial_ids[n_train:])
    train = [t for t in tuples if t["trial_id"] in train_set]
    eval_t = [t for t in tuples if t["trial_id"] in eval_set]
    return train, eval_t
