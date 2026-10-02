"""The agent interface, and the one agent that ships (7-A1).

`TutorialAgent` demonstrates 6-R6's three methods and usually fails to complete a course.
That is the whole specification: 7-A1 ships a tutorial-grade agent because a competent one
would become the presumed answer, and because 2-E10 aims the hidden set's difficulty at
the *internal* reference agent's completion rate -- so shipping that agent would hand
teams a policy calibrated against the set they are scored on.

The competent agents live in `usvnav.internal.agents`, which the participant bundle does
not contain (`usvnav.bundle`). Nothing in this module may import from there.

The interface, in full:

    class MyAgent:
        def __init__(self, config_path=None): ...   # 6-R6, points inside your bundle
        def reset(self, meta): ...                  # 4-V1's disclosure, once per episode
        def act(self, obs) -> [v_cmd, w_cmd]: ...   # every tick

No state may carry from one episode to the next (6-R8); `reset` is where it is cleared.
"""

from __future__ import annotations

import numpy as np

from .plant import V_CRUISE, W_MAX


class TutorialAgent:
    """Minimal interface demonstration (7-A1). Steers at the goal, ignores everything.

    Deliberately unable to avoid anything: X1 ran it over 324 courses and it completed
    none of them, every failure a static collision (§6.1). That is what makes it usable
    as the naive role in 3-K13's anti-triviality check as well as as the example.
    """

    def __init__(self, config_path=None):
        self.config_path = config_path

    def reset(self, meta):
        self.meta = meta

    def act(self, obs):
        _, bearing = obs["wp_polar"]
        return np.array([V_CRUISE * 0.6, float(np.clip(bearing, -W_MAX, W_MAX))])
