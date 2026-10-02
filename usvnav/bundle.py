"""What ships to participants, as an allow-list rather than a habit (7-A1, 7-A2, 7-A4).

Three things this file is, in order of how easy they are to get wrong.

**An allow-list.** `SHIPPED` names every path a team receives. `INTERNAL` names every path
they must not. `audit()` walks the tree and fails if a file is in neither, so adding a
module is a decision about which side it belongs on rather than a default. A deny-list
would have the opposite failure: a new internal file ships until someone remembers it.

**An import check.** The allow-list alone does not protect anything. If a shipped module
imports `usvnav.internal`, deleting that package from the bundle breaks the bundle, and
whoever notices will fix it by putting the package back. `audit()` therefore also parses
every shipped module and reports any import that reaches the internal side. That is the
check the pre-split tree failed: `tests/test_render.py`, `tests/test_collide.py` and
`tools/figures.py` all imported `ReferenceAgent`, and 4-V13 requires the palette test to
ship *with* the renderer -- so the internal agent was load-bearing inside the shipped set.

**A build.** `build(dest)` copies the allow-list, and `tests/internal/test_bundle.py`
then imports the result in a subprocess with nothing else on the path. Only running it
proves the first two are true.

Not settled here: the bundle **size cap** (6-R9) and the manifest's contest metadata.
Those are contest-operations values.
"""

from __future__ import annotations

import ast
import datetime
import pathlib
import shutil
import subprocess

from . import __version__

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Written into every build by `build()`: the kit's identity, for a team to quote. The kit
#: is frozen for the contest (follow-up item 15), but its documents were corrected more than
#: once before it froze, and "which kit do you have?" needs an answer from inside the kit.
STAMP = "VERSION.txt"

#: Everything a team receives. Directories are taken whole, minus `INTERNAL`.
#:
#: `tests/` ships because 4-V13 requires the palette general-position check to travel with
#: the renderer -- "editing a colour later cannot silently break it" is a promise about
#: the code teams hold, not only about ours. `tools/editor.html` ships under 7-A2.
#: `examples/` is the worked example submission (T1-RES-18), `docs/` the generated schema,
#: rules and runtime documents (T1-RES-04/05/14/15), and `sets/` the frozen public
#: practice set (T1-RES-03) -- all three are what §9.6 adds to the kit. `tools/studio.html`
#: is the studio's page (T1-RES-21); `usvnav/studio.py` serves it.
SHIPPED = (
    "pyproject.toml",
    "README.md",
    "usvnav/",
    "tests/",
    "tools/editor.html",
    "tools/studio.html",
    "tools/demo-course.json",
    "examples/",
    "docs/",
    "sets/",
)

#: Everything that must not leave this tree, each with the decision that keeps it here.
INTERNAL = {
    "usvnav/internal/": "7-A1 -- the competent agents; shipping one erases the "
                         "completion-rate discriminator and hands teams the instrument "
                         "2-E10 calibrates the hidden set against",
    "usvnav/generate.py": "7-A8 -- the course generator is the definition of the hidden set; "
                           "participants author courses in the editor",
    "usvnav/scenario.py": "7-A8 -- the situation taxonomy and placement mechanics",
    "usvnav/channel.py": "7-A8 -- the channel constructor (part of the generator)",
    "usvnav/lanegen.py": "7-A8 -- lane threading and lane measurement (part of the generator)",
    "usvnav/accel.py": "organiser-side acceleration of the scorer's simulator (INTERNAL.md, \"Simulator "
                       "acceleration\"); the kit runs the Python it ships, and the scorer proves the kernels "
                       "draw the same bytes before using them",
    "usvnav/_native": "the built native kernels behind usvnav/accel.py, one file per interpreter ABI",
    "accel/": "the Rust crate the kernels are built from, and its build script",
    "tools/accel_sweep.py": "the byte-identity sweep that qualifies a build of the kernels on a set",
    "INTERNAL.md": "organiser notes: the generator, the experiments, the tooling",
    "PATCHNOTES.md": "the participants' change log, published by the hub as patchnotes.html -- the page is the "
                     "copy participants read; the kit does not carry a snapshot that would go stale",
    "tests/internal/": "tests of the above, plus this bundle's own audit",
    "tools/x1.py": "X1, the difficulty sweep -- organizer-side (§6.1)",
    "tools/x1_summary.py": "X1's summary tables",
    "tools/x2.py": "X2, the item-spread field (§6)",
    "tools/x2_types.py": "X2 per situation type -- the evidence for §9.3's three changes",
    "tools/x3.py": "X3, the per-tick wall clock (§6)",
    "tools/x4.py": "X4, hidden-set jitter (§6)",
    "tools/x5.py": "X5, the 4-V11 paired drift check on the corrected plant (§9.7)",
    "tools/signoff.py": "evidence for the PROPOSED rows",
    "tools/figures.py": "figure generation for the decision log; drives the reference "
                        "agent, so it is internal by import as well as by purpose",
    "results/": "measurements, including per-course difficulty -- a difficulty label by "
                "another name (follow-up item 6)",
    "dist/": "the built bundle itself",
    ".venv/": "",
    ".git/": "",
}

#: Build products, which are neither shipped nor withheld -- they are regenerated.
#: Deliberately short. An earlier version also ignored `.json`, `.png` and `.txt`, which
#: would have let a results dump or a course file carrying a difficulty label sit unfiled
#: and never be noticed (follow-up item 6). Data is exactly the thing that has to be on
#: one side of the line on purpose, so only generated output is exempt.
_IGNORED = (".pyc",)

#: Directories a tool regenerates. `usvnav.egg-info/` appears the moment anyone runs
#: `pip install -e .`, so it is present in the main working copy and absent from a fresh
#: worktree -- which is why the audit passed in one and failed in the other. Since
#: 2026-09-15 the general rule is `_git_ignored()`: whatever `.gitignore` names is a build
#: product and is skipped; this tuple is the fallback for a tree without git.
_GENERATED = ("usvnav.egg-info/",)


def _git_ignored(root: pathlib.Path) -> frozenset[str]:
    """The files git ignores under `root` (relative, posix), or an empty set without git.

    The hub's checkout carries this tree as a subtree and installs it editable, which
    writes `usvnav.egg-info/` beside the package; after the 2026-09-15 rename the previous
    package's egg-info sat there too, and the audit refused the build for it. Those are
    build products, and `.gitignore` already says so once; the audit reads that instead
    of keeping its own list.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--ignored",
                              "--exclude-standard", "-z"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    return frozenset(p for p in out.stdout.split("\0") if p)


def _is_internal(rel: str) -> str | None:
    for prefix, why in INTERNAL.items():
        if rel == prefix or rel.startswith(prefix):
            return why
    return None


def _is_shipped(rel: str) -> bool:
    if _is_internal(rel):
        return False
    return any(rel == s or (s.endswith("/") and rel.startswith(s)) for s in SHIPPED)


def shipped_files(root: pathlib.Path = ROOT):
    """Every file a team receives, in sorted order. Git-ignored files never ship."""
    ignored = _git_ignored(root)
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_shipped(rel) and not rel.endswith(".pyc") and rel not in ignored:
            out.append(rel)
    return out


def _imports(path: pathlib.Path):
    """Every module name a file imports, including inside functions.

    `ast` rather than a regex because the pre-split tree hid three of the four offending
    imports *inside* test functions, where a top-of-file grep does not look.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative: resolve against the package
                pkg = path.relative_to(ROOT).parent.as_posix().replace("/", ".")
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                names.append(f"{base}.{node.module}" if node.module else base)
            else:
                names.append(node.module or "")
    return names


def _module_path(name: str) -> str | None:
    """Where a dotted module name would live in this tree, if it lives here at all."""
    parts = name.split(".")
    if parts[0] not in ("usvnav", "tools", "tests"):
        return None
    stem = "/".join(parts)
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if (ROOT / candidate).exists():
            return candidate
    return None


def audit(root: pathlib.Path = ROOT):
    """Return `(unclassified, leaks)`; both empty is the only acceptable result.

    A leak is any shipped module importing something in this tree that does not ship --
    stated that way rather than as "imports `usvnav.internal`", because the first version
    of this check was written the narrow way and missed `usvnav/cli.py`'s lazy
    `from tools.figures import figure_course`. `tools/figures.py` is internal by import
    (it drives the reference agent), so `usvnav view` -- a command the README tells
    participants to run -- was reaching across the line without ever naming
    `usvnav.internal`. The general form is the only one that catches that.
    """
    unclassified, leaks = [], []
    ignored = _git_ignored(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if (rel.startswith(".") or rel.endswith(_IGNORED)
                or rel.startswith(_GENERATED) or rel in ignored):
            continue
        if _is_internal(rel):
            continue
        if not _is_shipped(rel):
            unclassified.append(rel)
            continue
        if path.suffix == ".py":
            for name in _imports(path):
                target = _module_path(name)
                if target is not None and not _is_shipped(target):
                    leaks.append((rel, name))
    return unclassified, leaks


def _git(*args: str) -> str:
    """One git query against the tree, or "" when there is no git or no repository."""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def stamp_text(now: datetime.datetime | None = None) -> str:
    """The contents of `VERSION.txt`: version, commit, build time, and what to do with them."""
    commit = _git("rev-parse", "--short=12", "HEAD") or "unknown"
    dirty = " +uncommitted" if _git("status", "--porcelain", "--untracked-files=no") else ""
    when = (now or datetime.datetime.now(datetime.timezone.utc).astimezone()).isoformat(timespec="seconds")
    return (f"usvnav {__version__}\n"
            f"commit {commit}{dirty}\n"
            f"built {when}\n"
            "\n"
            "This is the build of the Track 1 kit you have. Quote the three lines above when you\n"
            "write to the organisers; the kit's zip checksum is published beside the download.\n")


def build(dest: pathlib.Path) -> pathlib.Path:
    """Copy the allow-list into `dest`, which is emptied first, and stamp it (`VERSION.txt`)."""
    dest = pathlib.Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    for rel in shipped_files():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (dest / STAMP).write_text(stamp_text(), encoding="utf-8")
    return dest


def cmd_bundle(args):
    unclassified, leaks = audit()
    for rel in unclassified:
        print(f"  [unfiled ] {rel}")
    for rel, name in leaks:
        print(f"  [leak    ] {rel} imports {name}")
    if unclassified or leaks:
        print("\nEvery file has to be on one side of 7-A1 and no shipped file may import")
        print("the internal side. Add it to SHIPPED or to INTERNAL in usvnav/bundle.py.")
        return 1
    files = shipped_files()
    print(f"{len(files)} files ship, {len(INTERNAL)} paths held back:")
    for prefix, why in sorted(INTERNAL.items()):
        if why:
            print(f"  {prefix:<22} {why}")
    if args.dest:
        out = build(args.dest)
        n = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"\nwrote {out}  ({n} files)")
    return 0
