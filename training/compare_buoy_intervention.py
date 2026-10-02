"""Compare existing normal/hidden-buoy rollout traces; no policy execution."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--normal", default="post-00")
    parser.add_argument("--hidden", default="hidden-00")
    args = parser.parse_args()
    output = args.run / "vision-intervention-comparison.json"
    if output.exists():
        parser.error("Refusing to overwrite intervention comparison")
    for phase in (args.normal, args.hidden):
        if Path(phase).name != phase:
            parser.error("Use plain phase names")
    read = lambda phase, suffix: json.loads((args.run / f"{phase}-{suffix}.json").read_text(encoding="utf-8"))
    normal_evaluation, hidden_evaluation = (read(phase, "evaluation") for phase in (args.normal, args.hidden))
    normal, hidden = ({row["tick"]: row for row in read(phase, "trace")} for phase in (args.normal, args.hidden))
    ticks = sorted(set(normal) & set(hidden))
    position_delta = np.asarray([np.linalg.norm(np.asarray(normal[t]["pose"][:2]) - hidden[t]["pose"][:2]) for t in ticks])
    residual_delta = np.asarray([np.linalg.norm(np.asarray(normal[t]["residual"]) - hidden[t]["residual"]) for t in ticks])
    changed = [tick for tick, delta in zip(ticks, residual_delta) if delta > 1e-7]
    summary = dict(normal=normal_evaluation, hidden_buoy=hidden_evaluation,
                   common_trace_ticks=len(ticks), max_common_tick_position_delta_m=float(position_delta.max()),
                   mean_common_tick_position_delta_m=float(position_delta.mean()),
                   max_common_tick_residual_delta=float(residual_delta.max()),
                   mean_common_tick_residual_delta=float(residual_delta.mean()),
                   first_residual_difference_tick=changed[0] if changed else None,
                   changed_residual_ticks=len(changed),
                   diagnostic_limit="A single paired vision intervention tests this rollout only. Later poses/public inputs may diverge after actions differ; outcomes do not establish generalized visual avoidance.")
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
