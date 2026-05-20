"""ad-hoc: B6 closed-loop trace inspection."""
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    r = json.load(f)
sel = list(r["selections"].values())[0]
print(f"trial: {r['trial_id']}")
print(f"target_score_before: {sel['target_score_before']:.4f}, after: {sel['target_score_after']:.4f}")
print(f"netgain: {sel['netgain_provisional']:+.4f}, fidelity_loss: {sel['fidelity_loss_protocol_a']:+.4f}")
print(f"tool_call_count: {sel['tool_call_count']}, rolled_back: {sel['rolled_back']}")
print(f"score_trace: {[round(x, 4) for x in sel['score_trace']]}")
print("decision_trace:")
for i, d in enumerate(sel["decision_trace"]):
    print(f"  step {i}: primary={d.get('metadata_primary_evaluator')} tool={d.get('selected_tool')} strength={d.get('strength')}")
print(f"per_evaluator before: {sel['per_evaluator_before_max']}")
print(f"per_evaluator after:  {sel['per_evaluator_after_max']}")
