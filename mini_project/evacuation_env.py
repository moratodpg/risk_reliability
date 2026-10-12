"""Educational crowd model: individual travel plus FIFO door queues.

Distances: m; times: s from ignition. This is a queue-based agent model, not
a validated pedestrian dynamics or fire model. No collision forces or smoke field.
Only NumPy is needed for simulation; rendering imports Matplotlib.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Distribution:
    """Positive marginals. Lognormal parameters are arithmetic mean and CV."""
    family: str
    mean: float
    cv: float

    def sample(self, rng, size=None):
        if self.mean <= 0 or self.cv < 0:
            raise ValueError("mean must be positive and CV non-negative")
        if self.cv == 0:
            return np.full(size, self.mean) if size is not None else self.mean
        if self.family == "lognormal":
            s = np.sqrt(np.log1p(self.cv**2))
            return rng.lognormal(np.log(self.mean) - s*s/2, s, size=size)
        if self.family == "gamma":
            return rng.gamma(1/self.cv**2, self.mean*self.cv**2, size=size)
        raise ValueError("Supported families: lognormal, gamma; extend sample for others")


@dataclass(frozen=True)
class InputModel:
    detection: Distribution
    response: Distribution
    speed: Distribution
    available_time: Distribution = Distribution("lognormal", 175.0, 0.20)
    flow_per_metre: Distribution = Distribution("lognormal", 1.20, 0.15)
    occupancy_min: int = 50
    occupancy_max: int = 90
    individual_delay_max: float = 12.0


@dataclass(frozen=True)
class Decision:
    name: str
    label: str
    exit_widths: tuple
    annual_cost: float
    response_factor: float = 1.0
    detection_factor: float = 1.0


DECISIONS = {
    "baseline": Decision("baseline", "Existing exit", (1.2, 0.0), 0.0),
    "training": Decision("training", "Evacuation training", (1.2, 0.0), 600.0, 0.70),
    "wider_exit": Decision("wider_exit", "Widen existing exit", (1.8, 0.0), 900.0),
    "second_exit": Decision("second_exit", "Add second exit", (1.2, 1.2), 1800.0),
}
ROOM = np.array([16.0, 12.0])
EXITS = np.array([[16.0, 3.0], [16.0, 9.0]])


@dataclass(frozen=True)
class Scenario:
    positions: np.ndarray
    speeds: np.ndarray
    individual_delays: np.ndarray
    detection: float
    response: float
    available_time: float
    flow_per_metre: float


def draw_scenario(rng, model):
    """One fire scenario: shared event variables and individual attributes.

    Shared detection/response/flow induces dependence among evacuees even
    though the underlying inputs are independent in the baseline model.
    Draw once, then reuse the Scenario across decisions for a paired comparison.
    """
    if not 1 <= model.occupancy_min <= model.occupancy_max:
        raise ValueError("Occupancy bounds must be positive and ordered")
    if model.individual_delay_max < 0:
        raise ValueError("individual_delay_max must be non-negative")
    n = int(rng.integers(model.occupancy_min, model.occupancy_max + 1))
    return Scenario(
        positions=rng.uniform([0.4, 0.4], ROOM - 0.4, size=(n, 2)),
        speeds=np.asarray(model.speed.sample(rng, n)),
        individual_delays=rng.uniform(0, model.individual_delay_max, n),
        detection=float(model.detection.sample(rng)),
        response=float(model.response.sample(rng)),
        available_time=float(model.available_time.sample(rng)),
        flow_per_metre=float(model.flow_per_metre.sample(rng)),
    )


def fifo_departures(arrival, headway):
    """Vectorized exact solution of e_i = max(a_i, e_(i-1)) + h.

    arrival must be sorted, non-negative; h is constant for this door/event.
    """
    arrival = np.asarray(arrival, dtype=float)
    if headway <= 0 or np.any(arrival < 0) or np.any(np.diff(arrival) < 0):
        raise ValueError("Need sorted non-negative arrivals and positive headway")
    i = np.arange(len(arrival))
    return np.maximum.accumulate(arrival - i*headway) + (i+1)*headway


def simulate(scenario, decision, trace=False):
    """Complete evacuation schedule without animation or time stepping.

    All people use the nearest open exit (Euclidean distance), then FIFO.
    A failure occurs if the last person finishes exiting at or after ASET.
    The schedule continues after ASET to define RSET; it does not model injury.
    """
    s, d = scenario, decision
    widths = np.asarray(d.exit_widths, dtype=float)
    if widths.shape != (len(EXITS),) or np.any(widths < 0):
        raise ValueError("Provide a non-negative width for each of the two exits")
    if d.response_factor <= 0 or d.detection_factor <= 0:
        raise ValueError("Time multipliers must be positive")
    active = np.flatnonzero(widths > 0)
    n = len(s.speeds)
    if not len(active):
        if trace:
            raise ValueError("Cannot animate a model with no open exit")
        return {"rset": np.inf, "aset": s.available_time, "g": -np.inf,
                "failed": True, "late_people": n, "occupancy": n}
    if n == 0 or np.any(s.speeds <= 0) or s.flow_per_metre <= 0:
        raise ValueError("Need people, positive speeds and positive flow")
    distance = np.linalg.norm(s.positions[:, None, :] - EXITS[active][None, :, :], axis=2)
    choice = np.argmin(distance, axis=1)
    assigned = active[choice]
    travel = distance[np.arange(n), choice] / s.speeds
    # Training affects the shared response AND the individual's residual delay.
    start = d.detection_factor*s.detection + d.response_factor*(s.response+s.individual_delays)
    arrival = start + travel
    finish = np.empty(n)
    headway = np.zeros(len(EXITS))
    queue_order = {}
    for j in active:
        ids = np.flatnonzero(assigned == j)
        ids = ids[np.argsort(arrival[ids], kind="stable")]
        headway[j] = 1.0/(s.flow_per_metre*widths[j])
        if len(ids):
            finish[ids] = fifo_departures(arrival[ids], headway[j])
        queue_order[j] = ids
    rset = float(finish.max())
    margin = float(s.available_time-rset)
    result = {"rset": rset, "aset": s.available_time, "g": margin,
              "failed": bool(margin <= 0),
              "late_people": int(np.sum(finish >= s.available_time)), "occupancy": n}
    if trace:
        result.update(scenario=s, decision=d, start=start, arrival=arrival,
                      finish=finish, assigned=assigned, headway=headway, queue_order=queue_order)
    return result


def monte_carlo(model, decisions=None, n=2000, seed=2026):
    """Independent events; common scenarios across alternatives, no rendering."""
    if n < 2:
        raise ValueError("At least two events are needed for uncertainty estimates")
    decisions = list(DECISIONS.values()) if decisions is None else list(decisions)
    if len({d.name for d in decisions}) != len(decisions):
        raise ValueError("Decision names must be unique")
    rng = np.random.default_rng(seed)
    fields = ("rset", "aset", "g", "failed", "late_people", "occupancy")
    results = {d.name: {f: np.empty(n, dtype=bool if f == "failed" else float)
                       for f in fields} for d in decisions}
    for i in range(n):
        scenario = draw_scenario(rng, model)
        for d in decisions:
            row = simulate(scenario, d)
            for f in fields:
                results[d.name][f][i] = row[f]
    return results


def probability_summary(failed):
    """Plug-in SE/CoV and Wilson 95% interval for fixed-N independent trials.

    SE is reported as NaN at 0 or N failures; the Wilson interval still works.
    """
    x = np.asarray(failed, dtype=bool)
    n = x.size
    if n < 2:
        raise ValueError("Need at least two observations")
    k = int(x.sum()); p = k/n; z = 1.959963984540054
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    se = np.sqrt(p*(1-p)/n) if 0 < k < n else np.nan
    return dict(N=n, failures=k, pf=p, se=se, cov=se/p if p > 0 else np.nan,
                ci_low=max(0.0, centre-half), ci_high=min(1.0, centre+half))


class EvacuationEnv:
    """RL-style viewer with reset/step/render, no Gym dependency or policy training.

    A design is chosen at reset; step(dt) advances observation time through the
    fixed simulated schedule. There is no per-step control action or reward.
    """
    def __init__(self, model):
        self.model = model
        self.record = None
        self.time = 0.0

    def reset(self, seed=7, decision=None):
        decision = DECISIONS["baseline"] if decision is None else decision
        s = draw_scenario(np.random.default_rng(seed), self.model)
        self.record = simulate(s, decision, trace=True)
        self.time = 0.0
        return self.observe()

    def step(self, dt=1.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.record is None:
            raise RuntimeError("Call reset first")
        self.time = min(self.time + dt, self.record["rset"])
        return self.observe()

    def observe(self):
        if self.record is None:
            raise RuntimeError("Call reset first")
        r, t = self.record, self.time
        status = np.full(len(r["finish"]), "waiting", dtype="U8")
        status[t >= r["start"]] = "walking"
        status[t >= r["arrival"]] = "queue"
        status[t >= r["finish"]] = "exited"
        fraction = np.clip((t-r["start"])/(r["arrival"]-r["start"]), 0, 1)
        pos = r["scenario"].positions + fraction[:, None]*(EXITS[r["assigned"]]-r["scenario"].positions)
        # Queue markers are schematic slots, NOT physical collision simulation.
        for j, ids in r["queue_order"].items():
            queued = ids[status[ids] == "queue"]
            rank = np.arange(len(queued))
            pos[queued, 0] = ROOM[0] - 0.45 - 0.45*(rank//5)
            pos[queued, 1] = EXITS[j, 1] + 0.38*((rank % 5)-2)
        return dict(time=t, positions=pos, status=status, exited=int(np.sum(status == "exited")),
                    done=bool(t >= r["rset"]), untenable=bool(t >= r["aset"]))

    def render(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.clear()
        obs, r = self.observe(), self.record
        ax.add_patch(Rectangle((0, 0), *ROOM, facecolor="#fff0ec" if obs["untenable"] else "#f1f5f9",
                               edgecolor="#334155", lw=2))
        for j, width in enumerate(r["decision"].exit_widths):
            if width > 0:
                ax.plot([ROOM[0]]*2, [EXITS[j, 1]-width/2, EXITS[j, 1]+width/2],
                        color="#059669", lw=7, solid_capstyle="butt")
                ax.text(ROOM[0]+0.3, EXITS[j, 1], f"Exit {j+1}\n{width:g} m", va="center", fontsize=9)
            else:
                ax.text(ROOM[0]+0.3, EXITS[j, 1], "No exit", va="center", fontsize=9, color="#64748b")
        colors = {"waiting": "#94a3b8", "walking": "#0065bd", "queue": "#d97706"}
        for state, color in colors.items():
            mask = obs["status"] == state
            ax.scatter(*obs["positions"][mask].T, s=32, c=color, label=f"{state}: {mask.sum()}",
                       edgecolors="white", linewidths=0.4)
        ax.set(xlim=(-0.5, 19.3), ylim=(-0.5, 12.5), xlabel="Room x [m]", ylabel="Room y [m]", aspect="equal")
        ax.set_title(f"{r['decision'].label} | t = {self.time:.0f} s | exited {obs['exited']}/{len(r['finish'])}\n"
                     f"ASET = {r['aset']:.0f} s | RSET = {r['rset']:.0f} s", fontsize=11)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
        ax.text(0.01, -0.26, "Queue slots are schematic. Travel paths have no collision avoidance.",
                transform=ax.transAxes, fontsize=8, color="#475569")
        return ax.figure
