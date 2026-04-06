import json
import os
from pathlib import Path


class StepLogger:
    """Structured per-step logger.

    Usage:
        logger = StepLogger("./outputs/experiment/run_name")
        logger.log(step=0, metrics={"reward": 0.3, "kl": 1.2, ...})
        logger.log_samples(step=0, samples=["Once upon a time..."])
        logger.save()               # writes metrics.jsonl to output_dir
        df = logger.to_dataframe()  # pandas DataFrame
        plot_training_curves(logger)
    """

    def __init__(self, output_dir: str = "./outputs"):
        self._records: list[dict] = []
        self._samples: list[tuple[int, list[str]]] = []
        self._output_dir = output_dir

    # ── Core logging ───────────────────────────────────────────────────────

    def log(self, step: int, metrics: dict) -> None:
        """Record a flat dict of metrics for this step."""
        self._records.append({"step": step, **metrics})

    def log_samples(self, step: int, samples: list[str]) -> None:
        """Record generated text samples at this step."""
        self._samples.append((step, samples))

    # ── Retrieval ──────────────────────────────────────────────────────────

    def get_metric(self, key: str) -> list:
        """Return all recorded values for a metric key, in step order."""
        return [r[key] for r in self._records if key in r]

    def last(self) -> dict:
        """Return the most recent step record."""
        return self._records[-1] if self._records else {}

    def to_dataframe(self):
        """Return all step records as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas")
        return pd.DataFrame(self._records)

    # ── Console output ─────────────────────────────────────────────────────

    def print_step(self, step: int, keys: list[str] | None = None) -> None:
        """Print the most recent record, optionally filtering to specific keys."""
        if not self._records:
            return
        rec = self._records[-1]
        if keys is not None:
            rec = {k: rec[k] for k in keys if k in rec}
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in rec.items() if k != "step"]
        print(f"  [step {step:4d}] " + " | ".join(parts))

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """Write all records as JSONL to output_dir/metrics.jsonl."""
        if path is None:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(self._output_dir, "metrics.jsonl")
        with open(path, "w") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")

    @classmethod
    def load(cls, path: str) -> "StepLogger":
        """Load a previously saved metrics.jsonl into a new StepLogger."""
        logger = cls(output_dir=str(Path(path).parent))
        with open(path) as f:
            for line in f:
                logger._records.append(json.loads(line))
        return logger
