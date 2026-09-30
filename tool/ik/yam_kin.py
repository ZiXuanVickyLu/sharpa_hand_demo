#!/usr/bin/env python
"""Fast kinematics + IK for one YAM arm with its Sharpa hand.

`ArmKin` lifts one arm's joint tree out of the Newton model (so it is, by
construction, the same robot the simulations and renders use) into plain
numpy: forward kinematics of all 43 links in ~0.3 ms and analytic Jacobians
for points and directions on any link. That makes a bounded, multi-start
least-squares IK practical -- over the 6 arm joints AND any of the 22 hand
joints at once -- where the finite-difference tracker in robot_scene.py
can only move the arm toward one target and has no joint-limit handling
beyond clipping.

Units are the rig's: centimeters, Z up, tabletop at z = 0.

    kin = ArmKin(bench, "left")
    q = kin.home()                      # 28 = 6 arm + 22 hand joints
    fk = kin.fk(q)                      # fk.R[b], fk.p[b] per link name/index
    sol = kin.solve(q, vars=kin.ARM, terms=[
        Point("thumb_fingertip", (0, 0, 0), target_cm, w=1.0),
        Axis("hand_wrist", local_axis, world_axis, w=5.0),
        Above("hand_wrist", (0, 0, 0), normal=(0, 0, 1), offset=6.0),
        Posture(q_ref, w=0.02)])

`python yam_kin.py` runs the self-test (numpy FK vs Newton FK, Jacobians vs
finite differences, hand mirror map).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

N_ARM = 6


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _xform(X7):
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(X7[3:7])
    T[:3, 3] = X7[:3]
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


@dataclass
class FK:
    R: np.ndarray        # (n_body, 3, 3) world rotation of every link
    p: np.ndarray        # (n_body, 3) world origin of every link (cm)
    axis: np.ndarray     # (n_q, 3) world axis of every revolute joint
    origin: np.ndarray   # (n_q, 3) world origin of every revolute joint

    def point(self, b, local):
        return self.p[b] + self.R[b] @ np.asarray(local, dtype=float)

    def direction(self, b, local):
        return self.R[b] @ np.asarray(local, dtype=float)


# ------------------------------------------------------------------ IK terms
@dataclass
class Point:
    """A point fixed on a link should reach a world target (cm)."""
    body: str | int
    local: tuple
    target: tuple
    w: float = 1.0
    mask: tuple = (1.0, 1.0, 1.0)  # per-axis weights, e.g. ignore y
    hard: bool = True  # False: a preference, left out of Solution.pos_err


@dataclass
class Axis:
    """A direction fixed on a link should align with a world direction."""
    body: str | int
    local: tuple
    world: tuple
    w: float = 1.0


@dataclass
class Above:
    """Keep a point on the positive side of a plane: n . x >= offset.
    One-sided (hinge): zero cost when satisfied."""
    body: str | int
    local: tuple
    normal: tuple
    offset: float
    w: float = 1.0


@dataclass
class Posture:
    """Pull the solved joints toward a reference (continuity / naturalness)."""
    q_ref: np.ndarray
    w: float | np.ndarray = 0.02


@dataclass
class Solution:
    q: np.ndarray
    cost: float
    pos_err: float       # worst error over the HARD Point terms, cm
    axis_err: float      # worst Axis error, degrees
    hinge: float         # worst Above violation, cm
    limit_margin: float  # smallest distance to a joint limit over vars, rad
    terms: dict = field(default_factory=dict)


class ArmKin:
    def __init__(self, bench, side: str):
        self.side = side
        arm = bench.left if side == "left" else bench.right
        self.arm = arm
        m = bench.model
        labels = [str(l).rsplit("/", 1)[-1] for l in m.body_label]
        parent = m.joint_parent.numpy()
        child = m.joint_child.numpy()
        jtype = m.joint_type.numpy()
        qstart = m.joint_q_start.numpy()
        axis = m.joint_axis.numpy()
        Xp = m.joint_X_p.numpy()
        Xc = m.joint_X_c.numpy()
        self.q0 = arm.dof_start
        self.n_q = arm.dof_count
        q_end = self.q0 + self.n_q

        # this arm's joints: the revolute ones own a coordinate in our range,
        # the fixed ones hang off a body we have already collected
        bodies, joints = {}, []
        for j in range(m.joint_count):
            is_rev = jtype[j] == 1
            mine = (self.q0 <= qstart[j] < q_end) if is_rev else (
                parent[j] in bodies or
                (parent[j] == -1 and f"{side}_" in labels[min(child[j] + 1,
                                                              len(labels) - 1)]))
            if not mine:
                continue
            b = len(bodies)
            bodies[int(child[j])] = b
            joints.append(dict(
                parent=bodies.get(int(parent[j]), -1), child=b,
                Tp=_xform(Xp[j]), Tci=_inv(_xform(Xc[j])),
                qi=int(qstart[j]) - self.q0 if is_rev else -1,
                axis=axis[int(m.joint_qd_start.numpy()[j])].astype(float)
                if is_rev else None))
        self.joints = joints
        self.global_body = {v: k for k, v in bodies.items()}  # ours -> model
        self.names = [None] * len(bodies)
        for gb, b in bodies.items():
            self.names[b] = labels[gb].replace(f"{side}_", "", 1)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n_body = len(bodies)

        # chain of coordinates that move each body
        self.chain = [[] for _ in range(self.n_body)]
        for jd in joints:
            up = list(self.chain[jd["parent"]]) if jd["parent"] >= 0 else []
            if jd["qi"] >= 0:
                up.append(jd["qi"])
            self.chain[jd["child"]] = up
        self.moves = np.zeros((self.n_body, self.n_q), dtype=bool)
        for b, ch in enumerate(self.chain):
            self.moves[b, ch] = True

        self.lo = m.joint_limit_lower.numpy()[self.q0:q_end].astype(float)
        self.hi = m.joint_limit_upper.numpy()[self.q0:q_end].astype(float)
        self.joint_names = ([f"arm{k + 1}" for k in range(N_ARM)]
                            + [n.replace(f"{side}_", "", 1)
                               for n in arm.finger_joint_names])
        self.jidx = {n: i for i, n in enumerate(self.joint_names)}
        self.ARM = list(range(N_ARM))
        self.THUMB = [i for i, n in enumerate(self.joint_names) if "thumb" in n]
        self.HAND = list(range(N_ARM, self.n_q))

    # ------------------------------------------------------------- basics
    def b(self, body):
        return body if isinstance(body, (int, np.integer)) else self.index[body]

    def home(self):
        return np.zeros(self.n_q)

    def clip(self, q, margin=0.0):
        return np.clip(q, self.lo + margin, self.hi - margin)

    def fk(self, q) -> FK:
        R = np.empty((self.n_body, 3, 3))
        p = np.empty((self.n_body, 3))
        ax = np.zeros((self.n_q, 3))
        og = np.zeros((self.n_q, 3))
        T = [None] * self.n_body
        for jd in self.joints:
            Tw = jd["Tp"] if jd["parent"] < 0 else T[jd["parent"]] @ jd["Tp"]
            if jd["qi"] >= 0:
                a = jd["axis"]
                ax[jd["qi"]] = Tw[:3, :3] @ a
                og[jd["qi"]] = Tw[:3, 3]
                th = q[jd["qi"]]
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]],
                              [-a[1], a[0], 0]])
                Rq = np.eye(4)
                Rq[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
                Tw = Tw @ Rq
            Tc = Tw @ jd["Tci"]
            T[jd["child"]] = Tc
            R[jd["child"]] = Tc[:3, :3]
            p[jd["child"]] = Tc[:3, 3]
        return FK(R, p, ax, og)

    def jac_point(self, fk: FK, b, x_world):
        """d(x_world)/dq for a point riding on link b -> (3, n_q)."""
        J = np.cross(fk.axis, x_world[None, :] - fk.origin).T
        return J * self.moves[b][None, :]

    def jac_dir(self, fk: FK, b, d_world):
        J = np.cross(fk.axis, d_world[None, :]).T
        return J * self.moves[b][None, :]

    # ----------------------------------------------------------------- IK
    def _residual(self, q, terms, want_jac):
        fk = self.fk(q)
        rs, Js = [], []
        for t in terms:
            if isinstance(t, Point):
                b = self.b(t.body)
                x = fk.point(b, t.local)
                wm = t.w * np.asarray(t.mask, dtype=float)
                rs.append(wm * (x - np.asarray(t.target, dtype=float)))
                if want_jac:
                    Js.append(wm[:, None] * self.jac_point(fk, b, x))
            elif isinstance(t, Axis):
                b = self.b(t.body)
                d = fk.direction(b, t.local)
                rs.append(t.w * (d - np.asarray(t.world, dtype=float)))
                if want_jac:
                    Js.append(t.w * self.jac_dir(fk, b, d))
            elif isinstance(t, Above):
                b = self.b(t.body)
                x = fk.point(b, t.local)
                n = np.asarray(t.normal, dtype=float)
                gap = float(n @ x) - t.offset
                active = gap < 0.0
                rs.append(np.array([t.w * gap if active else 0.0]))
                if want_jac:
                    Jp = self.jac_point(fk, b, x)
                    Js.append((t.w * (n @ Jp))[None, :] if active
                              else np.zeros((1, self.n_q)))
            elif isinstance(t, Posture):
                w = np.broadcast_to(np.asarray(t.w, dtype=float), (self.n_q,))
                rs.append(w * (q - t.q_ref))
                if want_jac:
                    Js.append(np.diag(w))
        r = np.concatenate(rs)
        return (r, np.vstack(Js)) if want_jac else r

    def solve(self, q_init, vars, terms, ties=(), margin=0.02,
              max_nfev=200) -> Solution:
        """Bounded least squares over the joints listed in `vars`; every other
        joint stays at its q_init value, except tied joints: each
        (dst, src, offset[, scale]) in `ties` slaves q[dst] = scale * q[src]
        + offset (e.g. a finger's DIP following its PIP), with the dependency
        carried through the Jacobian."""
        vars = np.asarray(vars, dtype=int)
        q_base = np.array(q_init, dtype=float)
        lo, hi = self.lo[vars] + margin, self.hi[vars] - margin
        ties = [(self.jidx[t[0]] if isinstance(t[0], str) else t[0],
                 self.jidx[t[1]] if isinstance(t[1], str) else t[1], t[2],
                 t[3] if len(t) > 3 else 1.0) for t in ties]

        def full(z):
            q = q_base.copy()
            q[vars] = z
            for d, s_, off, sc in ties:
                q[d] = np.clip(sc * q[s_] + off, self.lo[d] + margin,
                               self.hi[d] - margin)
            return q

        def jac(z):
            q = full(z)
            J = self._residual(q, terms, True)[1].copy()
            for d, s_, off, sc in ties:
                # a tied joint sitting on its clip does not follow its source
                if self.lo[d] + margin < sc * q[s_] + off < self.hi[d] - margin:
                    J[:, s_] += sc * J[:, d]
            return J[:, vars]

        res = least_squares(
            lambda z: self._residual(full(z), terms, False),
            np.clip(q_base[vars], lo, hi), jac=jac,
            bounds=(lo, hi), method="trf", max_nfev=max_nfev,
            xtol=1e-10, ftol=1e-12, gtol=1e-10)
        q = full(res.x)
        return self.report(q, terms, vars, cost=float(res.cost))

    def report(self, q, terms, vars=None, cost=0.0) -> Solution:
        fk = self.fk(q)
        pe = ae = hg = 0.0
        detail = {}
        for k, t in enumerate(terms):
            if isinstance(t, Point):
                e = (fk.point(self.b(t.body), t.local)
                     - np.asarray(t.target, float)) * (np.asarray(t.mask) > 0)
                if t.hard:
                    pe = max(pe, float(np.linalg.norm(e)))
                detail[f"point{k}:{t.body}"] = float(np.linalg.norm(e))
            elif isinstance(t, Axis):
                d = fk.direction(self.b(t.body), t.local)
                c = float(np.clip(d @ np.asarray(t.world, float)
                                  / (np.linalg.norm(d) * np.linalg.norm(t.world)),
                                  -1, 1))
                ae = max(ae, float(np.degrees(np.arccos(c))))
                detail[f"axis{k}:{t.body}"] = float(np.degrees(np.arccos(c)))
            elif isinstance(t, Above):
                gap = float(np.asarray(t.normal, float)
                            @ fk.point(self.b(t.body), t.local)) - t.offset
                hg = max(hg, -min(gap, 0.0))
        v = np.arange(self.n_q) if vars is None else np.asarray(vars)
        lm = float(np.minimum(q[v] - self.lo[v], self.hi[v] - q[v]).min())
        return Solution(q, cost, pe, ae, hg, lm, detail)

    def hand_axes(self):
        """Wrist-local unit axes of the open hand: `fingers` (wrist ->
        fingertips), `palm` (out of the palm, the way the fingers flex) and
        `thumb_side` (pinky -> index). Measured, not assumed: the left and
        right hand models are mirror images."""
        q = self.home()
        fk = self.fk(q)
        w = self.b("hand_wrist")
        Rw = fk.R[w]
        f = Rw.T @ (fk.p[self.b("middle_fingertip")] - fk.p[self.b("middle_PP")])
        f /= np.linalg.norm(f)
        q2 = q.copy()
        q2[self.jidx["middle_MCP_FE"]] = 0.4
        n = Rw.T @ (self.fk(q2).p[self.b("middle_fingertip")]
                    - fk.p[self.b("middle_fingertip")])
        n -= f * (n @ f)
        n /= np.linalg.norm(n)
        lat = Rw.T @ (fk.p[self.b("index_PP")] - fk.p[self.b("pinky_PP")])
        lat -= f * (lat @ f) + n * (lat @ n)
        lat /= np.linalg.norm(lat)
        return dict(fingers=f, palm=n, thumb_side=lat)

    def solve_multi(self, q_init, vars, terms, n_starts=24, seed=0,
                    spread=1.0, accept=None, **kw) -> list[Solution]:
        """Multi-start: q_init first, then random restarts of `vars` drawn
        uniformly within the limits (scaled by `spread` around q_init for
        spread < 1). Returns all solutions sorted by cost; `accept(sol)`
        filters them."""
        rng = np.random.default_rng(seed)
        vars = np.asarray(vars, dtype=int)
        sols = []
        for k in range(n_starts):
            q0 = np.array(q_init, dtype=float)
            if k > 0:
                u = rng.uniform(self.lo[vars], self.hi[vars])
                q0[vars] = (1 - spread) * q0[vars] + spread * u
            s = self.solve(q0, vars, terms, **kw)
            if accept is None or accept(s):
                sols.append(s)
        return sorted(sols, key=lambda s: s.cost)


# ------------------------------------------------------------ collision proxies
class Proxies:
    """Covering spheres for every link, fitted to the link's render meshes
    (k-means clusters about `spacing` cm apart, radius = farthest assigned
    vertex, so the union is a conservative cover that hugs long links). Used to measure table and arm-to-arm clearance of a
    solved pose -- the rig has no collision geometry of its own."""

    def __init__(self, bench, kin: ArmKin, spacing=2.5, seed=0):
        from scipy.cluster.vq import kmeans2
        m = bench.model
        sb = m.shape_body.numpy()
        st = m.shape_transform.numpy()
        mine = {gb: b for b, gb in kin.global_body.items()}
        body, center, radius = [], [], []
        self.verts = {}   # link -> mesh vertices in the link frame (exact checks)
        for i, src in enumerate(m.shape_source):
            if src is None or not hasattr(src, "vertices") or int(sb[i]) not in mine:
                continue
            T = _xform(st[i])
            v = np.asarray(src.vertices, dtype=float) @ T[:3, :3].T + T[:3, 3]
            b_ = mine[int(sb[i])]
            self.verts[b_] = (v if b_ not in self.verts
                              else np.vstack([self.verts[b_], v]))
            ext = v.max(0) - v.min(0)
            k = int(np.clip(np.prod(np.maximum(ext / spacing, 1.0)), 1,
                            min(48, max(1, len(v) // 30))))
            c, lab = kmeans2(v, k, minit="++", seed=seed)
            for j in range(k):
                pts = v[lab == j]
                if len(pts) == 0:
                    continue
                cj = 0.5 * (pts.min(0) + pts.max(0))
                body.append(mine[int(sb[i])])
                center.append(cj)
                radius.append(np.linalg.norm(pts - cj, axis=1).max())
        self.kin = kin
        self.body = np.array(body)
        self.center = np.array(center)
        self.radius = np.array(radius)
        hand0 = kin.b("hand_flange")
        self.is_hand = self.body >= hand0

    def world(self, fk: FK):
        return (np.einsum("nij,nj->ni", fk.R[self.body], self.center)
                + fk.p[self.body])

    def mesh_points(self, fk: FK, from_body="link2", stride=1):
        """World positions of the actual mesh vertices of every link from
        `from_body` outward (the base and link1 stand on the rail plate)."""
        b0 = self.kin.b(from_body)
        return np.vstack([v[::stride] @ fk.R[b].T + fk.p[b]
                          for b, v in self.verts.items() if b >= b0])

    def exact_table(self, fk: FK):
        """Height of the lowest mesh vertex above the table (z = 0), over
        the links that can get there: forearm, wrist and hand."""
        return float(self.mesh_points(fk, "link4")[:, 2].min())

    def table_clearance(self, fk: FK, skip_base=True):
        """Lowest sphere bottom above z = 0 (links that stand on the rail
        plate by design are skipped)."""
        c = self.world(fk)
        keep = self.body > self.kin.b("link1") if skip_base else slice(None)
        return float((c[keep, 2] - self.radius[keep]).min())


def exact_gap(pa: Proxies, fka: FK, pb: Proxies, fkb: FK, stride=4):
    """True smallest vertex-to-vertex distance (cm) between two arms, from
    link4 outward (forearm, wrist, hand)."""
    from scipy.spatial import cKDTree
    a = pa.mesh_points(fka, "link4", stride)
    b = pb.mesh_points(fkb, "link4", stride)
    return float(cKDTree(a).query(b)[0].min())


def pair_clearance(pa: Proxies, fka: FK, pb: Proxies, fkb: FK):
    """Smallest gap (cm) between any sphere of arm a and any of arm b."""
    ca, cb = pa.world(fka), pb.world(fkb)
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    return float((d - pa.radius[:, None] - pb.radius[None, :]).min())


def hand_mirror_signs(kin_l: ArmKin, kin_r: ArmKin):
    """Sign vector s (22,) with q_hand_right = s * q_hand_left giving the
    mirror-image hand posture. The two YAM arms are IDENTICAL (same
    handedness, translated) -- only the hands are mirrored -- so there is no
    exact joint-space mirror for the arm: mirror the task-space targets and
    solve each arm's IK instead. For the hand the map is exact; it is probed
    joint by joint in the wrist frame."""
    wl, wr = kin_l.b("hand_wrist"), kin_r.b("hand_wrist")
    Mx = np.diag([-1.0, 1.0, 1.0])
    hand = np.arange(wl, kin_l.n_body)  # links from the wrist outward
    s = np.ones(kin_l.n_q - N_ARM)
    for k in range(N_ARM, kin_l.n_q):
        errs = []
        for sign in (+1.0, -1.0):
            e = 0.0
            for amp in (0.25, -0.15):
                ql = np.zeros(kin_l.n_q)
                ql[k] = np.clip(amp, kin_l.lo[k], kin_l.hi[k])
                qr = np.zeros(kin_r.n_q)
                qr[k] = np.clip(sign * ql[k], kin_r.lo[k], kin_r.hi[k])
                fl, fr = kin_l.fk(ql), kin_r.fk(qr)
                e += np.abs((fl.p[hand] - fl.p[wl]) @ Mx
                            - (fr.p[hand] - fr.p[wr])).max()
            errs.append(e)
        s[k - N_ARM] = +1.0 if errs[0] <= errs[1] else -1.0
    return s


if __name__ == "__main__":
    import time

    from robot_scene import YamSharpaBench

    bench = YamSharpaBench(device="cpu", with_table=False)
    kl, kr = ArmKin(bench, "left"), ArmKin(bench, "right")
    print(f"left: {kl.n_body} links, {kl.n_q} joints; "
          f"right: {kr.n_body} links, {kr.n_q} joints")
    rng = np.random.default_rng(0)
    worst = 0.0
    for kin in (kl, kr):
        for _ in range(20):
            q = rng.uniform(kin.lo, kin.hi)
            bench.joint_q[:] = 0.0
            bench.joint_q[kin.q0:kin.q0 + kin.n_q] = q
            ref = np.asarray(bench._fk())
            fk = kin.fk(q)
            for b in range(kin.n_body):
                gb = kin.global_body[b]
                worst = max(worst, np.abs(fk.p[b] - ref[gb, :3]).max())
                worst = max(worst, np.abs(fk.R[b] - _quat_to_R(ref[gb, 3:7])).max())
    print(f"numpy FK vs Newton FK, 40 random poses: max |diff| = {worst:.2e}")

    q = rng.uniform(kl.lo, kl.hi)
    fk = kl.fk(q)
    b = kl.b("thumb_fingertip")
    x = fk.point(b, (0.3, -0.2, 0.5))
    J = kl.jac_point(fk, b, x)
    Jn = np.zeros_like(J)
    for k in range(kl.n_q):
        dq = np.zeros(kl.n_q); dq[k] = 1e-6
        Jn[:, k] = (kl.fk(q + dq).point(b, (0.3, -0.2, 0.5)) - x) / 1e-6
    print(f"analytic vs FD point Jacobian: max |diff| = {np.abs(J - Jn).max():.2e}")

    t0 = time.time()
    for _ in range(200):
        kl.fk(q)
    print(f"fk: {(time.time() - t0) / 200 * 1e3:.2f} ms")

    s = hand_mirror_signs(kl, kr)
    print("hand mirror signs:", {n: int(v) for n, v in
                                 zip(kl.joint_names[N_ARM:], s) if v < 0}
          or "all +1 (same joint values give the mirrored posture)")
    ql = np.zeros(kl.n_q)
    ql[N_ARM:] = rng.uniform(kl.lo[N_ARM:], kl.hi[N_ARM:])
    qr = ql.copy()
    qr[N_ARM:] *= s
    fl, fr = kl.fk(ql), kr.fk(qr)
    wl, wr = kl.b("hand_wrist"), kr.b("hand_wrist")
    err = np.abs((fl.p[wl:] - fl.p[wl]) @ np.diag([-1.0, 1, 1])
                 - (fr.p[wl:] - fr.p[wr])).max()
    print(f"hand mirror check at a random hand posture: {err:.2e} cm")

    pl, pr = Proxies(bench, kl), Proxies(bench, kr)
    f0l, f0r = kl.fk(kl.home()), kr.fk(kr.home())
    print(f"proxies: {len(pl.radius)} spheres per arm, radius "
          f"{pl.radius.min():.1f}..{pl.radius.max():.1f} cm; home pose: table "
          f"clearance {pl.table_clearance(f0l):.1f} cm, arm-to-arm "
          f"{pair_clearance(pl, f0l, pr, f0r):.1f} cm")
