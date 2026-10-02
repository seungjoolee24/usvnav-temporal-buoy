"""The scoring runtime, as data (T1-RES-15; 6-R1, 6-R10, 6-R12).

What a submission meets at scoring time is a fixed environment, and 6-R9 has the
submission's manifest name the environment it expects so the validator can check the two
agree. For the prototype there is exactly one runtime, and it is this dictionary. The
rules document (`docs/RULES.md`), the runtime document (`docs/RUNTIME.md`) and the
packaging spec render it rather than restating it, so the numbers have one home.

**Pinned, not read.** `RUNTIME` declares the interpreter and the package versions; a test
(`tests/internal/test_runtime.py`) asserts the scoring machine matches. Reading them off
the running interpreter would make the definition whatever happened to be installed,
which is the opposite of a definition, and follow-up item 15 wants the environment frozen
for the contest's duration -- so this is the thing that gets frozen.

**numpy and onnxruntime, nothing else.** The package ships with numpy as its one
dependency and the scoring machine has no network (6-R1), so anything else a submission
needs has to be inside its directory. The one addition is 6-R12 (owner, 2026-09-10): the
runtime provides `onnxruntime`, GPU build, pinned -- framework-neutral, installed once, and
the same version Track 3 pins, so the two tracks share one inference runtime. The CUDA
libraries it loads are pip packages too (`CUDA`), pinned for the same reason: the scoring
machine has no system CUDA, and "GPU build" means nothing unless the libraries under it are
part of the definition. The versions are the ones installed on the scoring machine on
2026-09-10; changing any of them is a change to the runtime, so it is a decision.
"""

from __future__ import annotations

import importlib.metadata as _md
import platform
import sys

RUNTIME_ID = "usvnav-runtime/1"

#: Distribution names (what `pip install` takes), pinned exactly (follow-up item 15).
PACKAGES = {
    "numpy": "2.5.3",
    "onnxruntime-gpu": "1.23.2",            # 6-R12; imports as `onnxruntime`
}

#: The CUDA libraries onnxruntime-gpu 1.23 loads, as the pip packages they ship in.
CUDA = {
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cudnn-cu12": "9.25.1.1",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-curand-cu12": "10.3.10.19",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "nvidia-nvjitlink-cu12": "12.9.86",
}

#: What `onnxruntime.get_available_providers()` must contain. The GPU build reports the
#: CUDA provider whether or not a card is present; `gpu_smoke` is what proves the card.
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

RUNTIME = {
    "id": RUNTIME_ID,
    "python": "3.13",                       # major.minor; the patch level is not pinned
    "packages": PACKAGES,
    "cuda": CUDA,
    "providers": PROVIDERS,
    "network": False,                       # 6-R1
    "gpu": "a submission must fit in 8 GB (a 4060-class budget, 6-R1); the scoring machine's "
           "card is a 16 GB 4080-class shared between two submissions at a time; onnxruntime's "
           "CUDAExecutionProvider is how a submission reaches it",
    "submission_form": "directory (6-R10)",
    "agent_process": "a child of the runner, observation and action over a local pipe "
                     "(6-R5); one process pair per (team, condition), episodes run "
                     "sequentially with reset() between them (6-R7b)",
    "wall_clock": "not real time -- the runner blocks on act() (6-R7); the per-tick cap "
                  "(6-R3) is a throughput guard of 220 ms (25x the measured 8.8 ms "
                  "floor): a late tick applies the previous action, more than 10 late ticks "
                  "in a row or 50 caps of lateness in an episode fail it as time_overrun (6-R2)",
}


def _installed(dist: str) -> str | None:
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


def check_environment() -> list[str]:
    """Where the running interpreter differs from `RUNTIME`. Empty means it matches."""
    out = []
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if have != RUNTIME["python"]:
        out.append(f"python {have}, runtime pins {RUNTIME['python']}")
    for dist, want in {**PACKAGES, **CUDA}.items():
        got = _installed(dist)
        if got is None:
            out.append(f"{dist} is not installed")
        elif got != want:
            out.append(f"{dist} {got}, runtime pins {want}")
    try:
        import onnxruntime as ort
    except ImportError:
        out.append("onnxruntime is not importable")
    else:
        missing = [p for p in PROVIDERS if p not in ort.get_available_providers()]
        if missing:
            out.append(f"onnxruntime lacks {', '.join(missing)} -- not the GPU build?")
    return out


def gpu_smoke(model_path) -> str:
    """Run `model_path` once on the CUDA provider and return the provider that ran it.

    Raises whatever onnxruntime raises when the card, the driver or the CUDA libraries are
    not there. This is the runtime's own check that "GPU build" is more than a wheel name;
    `tests/internal/test_runtime.py` runs it on a four-element Add.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    used = sess.get_providers()[0]
    if used != "CUDAExecutionProvider":
        raise RuntimeError(f"onnxruntime fell back to {used}")
    name = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in name.shape]
    sess.run(None, {name.name: np.ones(shape, dtype=np.float32)})
    return used


def describe() -> str:
    lines = [f"runtime {RUNTIME['id']}",
             f"  python      {RUNTIME['python']}  ({platform.python_implementation()})"]
    for name, ver in PACKAGES.items():
        lines.append(f"  {name:<11} {ver}")
    short = {k.removeprefix("nvidia-").removesuffix("-cu12"): v for k, v in CUDA.items()}
    lines.append("  cuda        " + ", ".join(f"{k} {v}" for k, v in short.items()))
    lines.append(f"  network     {'none' if not RUNTIME['network'] else 'yes'}")
    lines.append(f"  gpu         {RUNTIME['gpu']}")
    lines.append(f"  submission  {RUNTIME['submission_form']}")
    return "\n".join(lines)
