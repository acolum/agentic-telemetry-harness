"""Central configuration: models, prices, caps, study sizes.

Prices are USD per 1M tokens. Verify against https://openai.com/api/pricing
before a paid run; they are passed to the cost tracker and nothing else.
"""
from dataclasses import dataclass, field

PRICES_PER_M = {
    # model: (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "mock": (0.0, 0.0),
}

ARCHITECTURES = ["retrieval", "code", "multiagent"]

THREAT_CLASSES = [
    "recursive_prompt_mutation",
    "tool_output_poisoning",
    "latent_policy_exfiltration",
    "goal_drift",
    "confident_erroneous_action",
]

DETECTOR_TO_CLASS = {
    "prompt_graph_entropy": "recursive_prompt_mutation",
    "tool_call_provenance": "tool_output_poisoning",
    "outbound_token_anomaly": "latent_policy_exfiltration",
    "subgoal_objective_divergence": "goal_drift",
    "action_evidence_mismatch": "confident_erroneous_action",
}

RESPONSE_RUNGS = ["in_loop_clarification", "human_confirm", "capability_reduction", "full_halt"]

# Tools whose invocation is an irreversible side effect ("action emit").
ACTION_TOOLS = {"send_email", "http_post", "pay_invoice", "book_slot", "reply_booking", "write_file"}
OUTBOUND_TOOLS = {"send_email", "http_post"}


@dataclass
class StudyConfig:
    model: str = "gpt-4.1-mini"
    embed_model: str = "text-embedding-3-small"
    judge_model: str = ""              # for the perimeter output classifier; defaults to `model`
    cost_cap_usd: float = 80.0         # hard stop
    seed: int = 20261022
    out_dir: str = "runs/study"
    # sizes per architecture
    calibration_sessions: int = 80     # attack-free, used only to set thresholds
    eval_sessions: int = 500           # evaluation set per architecture
    attack_fraction: float = 0.6       # share of eval sessions that receive an injected condition
    max_steps: int = 12                # ReAct steps per agent
    worker_max_steps: int = 6          # per worker in the multi-agent planner
    max_subtasks: int = 3
    target_session_fpr: float = 0.05   # calibration target per detector per architecture
    enforce_policy: bool = True        # apply the response ladder (vs. observe-only)
    temperature: float = 0.7
    mock: bool = False
    classes: list = field(default_factory=lambda: list(THREAT_CLASSES))

    def __post_init__(self):
        if not self.judge_model:
            self.judge_model = self.model
