"""Newton-side robot bench: two YAM arms with Sharpa hands, bases at (+-0.2667, 0, 0.0254) m yawed 90 degrees,
tabletop z = 0. Builds the arms from the URDFs of asset/robot/yam_sharpa (scale 100, centimetres), replays a recorded
joint trajectory kinematically (eval_fk) and reports the hand links' collider spheres.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


import newton  # noqa: E402
from newton import ModelBuilder  # noqa: E402
import roborender_sim.robots as _robots  # noqa: E402
from roborender_sim.robots import add_yam_sharpa  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

# the URDFs of asset/robot/yam_sharpa; their meshes are in asset/robot/yam_sharpa/meshes
_robots.ASSETS_DIR = _Path(__file__).resolve().parents[2] / "asset" / "robot"      # asset/robot/yam_sharpa/

CM_TO_M = 0.01
RIG_BASE_X_CM = 26.67  # rig: 0.5334 m pitch
RIG_BASE_YAW_DEG = 90.0

# keywords selecting hand bodies that should carry collider spheres
_TIP_TOKENS = ("thumb", "index", "middle", "ring", "pinky")
_PALM_TOKENS = ("wrist_mount", "hand")


@dataclass
class SphereSpec:
    body: int
    local_offset_cm: np.ndarray  # in body frame
    radius_m: float
    label: str


class YamSharpaBench:
    def __init__(self, device="cuda:0", hand_side="left", with_table=False):
        self.device = device
        scene = ModelBuilder()
        if with_table:
            # tabletop (top at z=0), 0.762 x 1.270 m
            scene.add_shape_box(
                body=-1,
                xform=wp.transform((0.0, 30.0, -2.5), wp.quat_identity()),
                hx=63.5, hy=38.1, hz=2.5,
                cfg=ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            )
            scene.shape_color[-1] = wp.vec3(0.92, 0.90, 0.86)
            # rig rail plate: 0.910 x 0.260 x 0.0254 m, top face at z=+2.54 cm
            scene.add_shape_box(
                body=-1,
                xform=wp.transform((0.0, 0.0, 1.27), wp.quat_identity()),
                hx=45.5, hy=13.0, hz=1.27,
                cfg=ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            )
            scene.shape_color[-1] = wp.vec3(0.25, 0.26, 0.28)
        yaw = wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0), float(np.deg2rad(RIG_BASE_YAW_DEG))
        )
        BASE_Z_CM = 2.54  # bases stand on the rig rail-plate top
        self.left = add_yam_sharpa(
            scene, wp.transform((-RIG_BASE_X_CM, 0.0, BASE_Z_CM), yaw), side="left"
        )
        self.right = add_yam_sharpa(
            scene, wp.transform((+RIG_BASE_X_CM, 0.0, BASE_Z_CM), yaw), side="right"
        )
        self.body_labels = [str(l) for l in scene.body_label]
        self.model = scene.finalize(device=device)
        self.state = self.model.state()
        self.active = self.left if hand_side == "left" else self.right
        self.joint_q = self.model.joint_q.numpy().copy()
        self.sphere_specs = self._select_spheres(self.active)
        self._fk()
        self._anchor_pad_spheres()

    # ------------------------------------------------------------------ FK
    def _fk(self):
        jq = wp.array(self.joint_q, dtype=float, device=self.device)
        jqd = wp.zeros_like(self.model.joint_qd)
        newton.eval_fk(self.model, jq, jqd, self.state)
        return self.state.body_q.numpy()

    def set_frame(self, arm_q6: np.ndarray, finger_frac: float) -> np.ndarray:
        """Set active arm joints + interpolated finger posture; return body_q."""
        arm = self.active
        s = arm.dof_start
        n_arm = arm.arm_dof_count
        self.joint_q[s : s + n_arm] = arm_q6
        fo, fc = arm.finger_open_q, arm.finger_close_q
        self.joint_q[s + n_arm : s + arm.dof_count] = fo + finger_frac * (fc - fo)
        return self._fk()


    def set_fingers(self, finger_frac: float) -> None:
        arm = self.active
        s = arm.dof_start
        n_arm = arm.arm_dof_count
        fo, fc = arm.finger_open_q, arm.finger_close_q
        self.joint_q[s + n_arm : s + arm.dof_count] = fo + finger_frac * (fc - fo)


    def _anchor_pad_spheres(self):
        """Re-anchor distal-phalanx spheres to the closed-pad centroid.

        Body origins sit at joint frames, centimeters away from the actual
        finger pads. Measure once at the closed posture: express the TCP
        (closed-pad centroid, ee_offset) in each distal body frame and use
        that as the sphere's local offset -- at close the spheres then meet
        exactly at the pads, and they spread naturally as fingers open.
        """
        arm = self.active
        saved = self.joint_q.copy()
        self.set_fingers(1.0)
        bq = self._fk()
        X_ee = wp.transform(*bq[arm.ee_body])
        # anchor target = the PHYSICAL closed-pad centroid (mean of distal
        # body origins), never the baked-in ee_offset: after a mount change
        # that stale TCP sits ~14 cm off the pads, and spheres anchored to it
        # swing huge fast arcs mid-curl (slinging bystander objects)
        dp_origins = [wp.transform_get_translation(wp.transform(*bq[s.body]))
                      for s in self.sphere_specs if s.label.endswith("_dp")]
        cen = np.mean([[p[0], p[1], p[2]] for p in dp_origins], axis=0)
        p_tcp = wp.vec3(float(cen[0]), float(cen[1]), float(cen[2]))
        for spec in self.sphere_specs:
            short = spec.label
            if short.endswith("_dp") or short.endswith("_pp") or short.endswith("_mp"):
                X_b = wp.transform(*bq[spec.body])
                local = wp.transform_point(wp.transform_inverse(X_b), p_tcp)
                local = np.array([local[0], local[1], local[2]])
                if short.endswith("_dp"):
                    spec.local_offset_cm = local          # pad at TCP
                    spec.radius_m = 0.022
                else:
                    spec.local_offset_cm = 0.5 * local    # mid-phalanx toward pad
                    spec.radius_m = 0.016
        # re-measure the TCP for this mount: closed-pad centroid (mean of the
        # distal-phalanx bodies) expressed in the wrist frame — the URDF hand
        # mount changed, so the ArmHandle's baked-in tip offset is stale
        tips = []
        for spec in self.sphere_specs:
            if spec.label.endswith("_dp"):
                p = wp.transform_get_translation(wp.transform(*bq[spec.body]))
                tips.append([p[0], p[1], p[2]])
        if tips:
            centroid = np.mean(tips, axis=0)
            local = wp.transform_point(wp.transform_inverse(X_ee),
                                       wp.vec3(*centroid))
            arm.ee_offset = wp.transform([local[0], local[1], local[2]],
                                         wp.quat_identity())
        self.joint_q = saved
        self._fk()

    # ------------------------------------------------------- collider spheres
    def _select_spheres(self, arm) -> list[SphereSpec]:
        """One sphere per hand body (palm bigger, phalanges smaller)."""
        specs: list[SphereSpec] = []
        # bodies belonging to this arm's subtree: label prefix injected by
        # add_builder is f"a{b0}_"; recover the range from ee_body neighborhood
        # by walking labels with the same prefix as the ee body.
        ee_label = self.body_labels[arm.ee_body]
        prefix = ee_label.split("_", 1)[0] + "_"
        for bi, label in enumerate(self.body_labels):
            if not label.startswith(prefix):
                continue
            short = label.rsplit("/", 1)[-1].lower()
            if any(t in short for t in _TIP_TOKENS):
                r = 0.011 if ("dis" in short or "tip" in short) else 0.013
                specs.append(SphereSpec(bi, np.zeros(3), r, short))
            elif any(t in short for t in _PALM_TOKENS):
                specs.append(SphereSpec(bi, np.zeros(3), 0.028, short))
        if not specs:  # fallback: every body after the wrist in this subtree
            for bi in range(arm.ee_body, len(self.body_labels)):
                if self.body_labels[bi].startswith(prefix):
                    specs.append(SphereSpec(bi, np.zeros(3), 0.013,
                                            self.body_labels[bi]))
        return specs

    def sphere_state(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World sphere centers (rig meters) + radii for the current pose."""
        centers = np.empty((len(self.sphere_specs), 3), dtype=np.float64)
        radii = np.empty(len(self.sphere_specs), dtype=np.float64)
        for k, spec in enumerate(self.sphere_specs):
            X = wp.transform(*body_q[spec.body])
            p = wp.transform_point(X, wp.vec3(*spec.local_offset_cm))
            centers[k] = np.array([p[0], p[1], p[2]]) * CM_TO_M
            radii[k] = spec.radius_m
        return centers, radii

    def ee_tip_rig(self, body_q: np.ndarray) -> np.ndarray:
        arm = self.active
        X = wp.transform(*body_q[arm.ee_body])
        p = wp.transform_point(X, wp.transform_get_translation(arm.ee_offset))
        return np.array([p[0], p[1], p[2]]) * CM_TO_M


class EETracker:
    """Damped-least-squares IK: track an EE pose trajectory with the 6 arm DOFs.

    Finite-difference Jacobian through Newton's eval_fk (kinematic tracking, a
    few iterations per control frame). Error is 6D: position (cm) + rotation
    vector of R_target @ R_fk^T.
    """

    def __init__(self, bench: YamSharpaBench, iters=10, damping=1e-2,
                 step_clamp=0.25, fd_eps=1e-3, arm=None, track_wrist=False):
        self.b = bench
        self.arm = arm if arm is not None else bench.active
        self.track_wrist = track_wrist
        self.iters = iters
        self.damping = damping
        self.step_clamp = step_clamp
        self.fd_eps = fd_eps
        arm = self.arm
        self.sl = slice(arm.dof_start, arm.dof_start + arm.arm_dof_count)
        model = bench.model
        self.lo = model.joint_limit_lower.numpy()[self.sl] + 0.03
        self.hi = model.joint_limit_upper.numpy()[self.sl] - 0.03
        # optional terminal override: track any link along the chain, moving
        # only the joints upstream of it (set via set_terminal)
        self.term_body = None
        self.term_offset = wp.transform_identity()
        self.n_active = arm.arm_dof_count

    def set_terminal(self, body=None, offset=None, n_active=None):
        """Choose which link the tracker moves ("terminal") and how many
        leading arm joints participate. body=None restores the default
        end-effector (pad-centroid TCP, all joints)."""
        self.term_body = body
        self.term_offset = offset if offset is not None else wp.transform_identity()
        self.n_active = self.arm.arm_dof_count if n_active is None else n_active

    def _ee_pose(self, body_q):
        arm = self.arm
        if self.term_body is not None:
            X = wp.transform(*body_q[self.term_body]) * self.term_offset
            p = np.array([*wp.transform_get_translation(X)])  # cm
            R = np.array(wp.quat_to_matrix(wp.transform_get_rotation(X))).reshape(3, 3)
            return p, R
        off = wp.transform_identity() if self.track_wrist else arm.ee_offset
        X = wp.transform(*body_q[arm.ee_body]) * off
        p = np.array([*wp.transform_get_translation(X)])  # cm
        R = np.array(wp.quat_to_matrix(wp.transform_get_rotation(X))).reshape(3, 3)
        return p, R

    @staticmethod
    def _rotvec(R):
        cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        ang = np.arccos(cos)
        if ang < 1e-8:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return axis / (2.0 * np.sin(ang)) * ang

    def _error(self, body_q, p_des_cm, R_des, axis_pair=None):
        p, R = self._ee_pose(body_q)
        if axis_pair is not None:
            # align wrist axes with world directions. Difference form, NOT
            # the cross product: the cross has zero gradient at exactly
            # 180 deg, so a flipped palm never turns. One pair leaves yaw
            # free; a list of pairs pins the full orientation.
            pairs = axis_pair if isinstance(axis_pair, list) else [axis_pair]
            rot = np.concatenate([
                np.asarray(a_world, dtype=np.float64)
                - R @ np.asarray(a_local, dtype=np.float64)
                for a_local, a_world in pairs])
        else:
            rot = self._rotvec(R_des @ R.T)
        return np.concatenate([p_des_cm - p, rot])

    def track(self, p_des_cm, R_des, rot_weight=8.0, axis_pair=None):
        """Iterate IK toward the target; returns final body_q and error norm."""
        b = self.b
        for _ in range(self.iters):
            q0 = b.joint_q[self.sl].copy()
            bq = b._fk()
            err = self._error(bq, p_des_cm, R_des, axis_pair)
            err_w = err.copy()
            err_w[3:] *= rot_weight
            J = np.zeros((len(err), 6))
            for j in range(6):
                if j >= self.n_active:
                    continue  # joint downstream of the terminal: frozen
                q_pert = q0.copy()
                q_pert[j] += self.fd_eps
                b.joint_q[self.sl] = q_pert
                bq_p = b._fk()
                e_p = self._error(bq_p, p_des_cm, R_des, axis_pair)
                e_p[3:] *= rot_weight
                J[:, j] = (err_w - e_p) / self.fd_eps  # = +d(fk)/dq for the position rows
            b.joint_q[self.sl] = q0
            JJt = J @ J.T + self.damping * np.eye(len(err))
            dq = J.T @ np.linalg.solve(JJt, err_w)
            dq = np.clip(dq, -self.step_clamp, self.step_clamp)
            b.joint_q[self.sl] = np.clip(q0 + dq, self.lo, self.hi)
        bq = b._fk()
        return bq, np.linalg.norm(self._error(bq, p_des_cm, R_des, axis_pair)[:3])


def quat_to_R(q: wp.quat) -> np.ndarray:
    return np.array(wp.quat_to_matrix(q)).reshape(3, 3)


# ------------------------------------------------------------ dual-arm helpers

def set_fingers_for(bench: YamSharpaBench, arm, frac: float) -> None:
    """Finger posture for a specific arm (bench.set_fingers only does .active)."""
    s = arm.dof_start
    n_arm = arm.arm_dof_count
    fo, fc = arm.finger_open_q, arm.finger_close_q
    bench.joint_q[s + n_arm : s + arm.dof_count] = fo + frac * (fc - fo)


def set_hand_groups(bench: YamSharpaBench, arm, fingers: float, thumb: float) -> None:
    """Independent thumb vs four-finger posture (scoop-and-clamp grasps)."""
    s = arm.dof_start
    n_arm = arm.arm_dof_count
    fo, fc = arm.finger_open_q, arm.finger_close_q
    is_thumb = np.array(["thumb" in n.lower() for n in arm.finger_joint_names])
    frac = np.where(is_thumb, thumb, fingers)
    bench.joint_q[s + n_arm : s + arm.dof_count] = fo + frac * (fc - fo)


def build_second_hand(bench: YamSharpaBench, arm) -> list:
    """Collider-sphere specs + pad anchoring for the non-active arm."""
    saved_active, saved_specs = bench.active, bench.sphere_specs
    bench.active, bench.sphere_specs = arm, bench._select_spheres(arm)
    bench._anchor_pad_spheres()
    specs = bench.sphere_specs
    bench.active, bench.sphere_specs = saved_active, saved_specs
    return specs


def sphere_state_for(specs, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World sphere centers (rig meters) + radii for an explicit spec list."""
    centers = np.empty((len(specs), 3), dtype=np.float64)
    radii = np.empty(len(specs), dtype=np.float64)
    for k, spec in enumerate(specs):
        X = wp.transform(*body_q[spec.body])
        p = wp.transform_point(X, wp.vec3(*spec.local_offset_cm))
        centers[k] = np.array([p[0], p[1], p[2]]) * CM_TO_M
        radii[k] = spec.radius_m
    return centers, radii


def track_with_table_clearance(ik: EETracker, bench: YamSharpaBench, specs,
                               target_m: np.ndarray, margin=0.005, tries=6,
                               rot_weight=0.0, axis_pair=None):
    """IK with non-penetration by constraint projection (Method B).

    Solve, FK the collider spheres, and if any sphere dips below
    `margin` above the table (rig z=0), re-solve with the target raised by
    the deficit. If it still violates after `tries`, restore the previous
    joint state — a penetrating pose is never returned.
    Returns (body_q, ik_err_cm, clearance_m).
    """
    q_prev = bench.joint_q.copy()
    t = np.asarray(target_m, dtype=np.float64).copy()
    for _ in range(tries):
        body_q, err = ik.track(t * 100.0, np.eye(3), rot_weight=rot_weight,
                               axis_pair=axis_pair)
        centers, radii = sphere_state_for(specs, body_q)
        c = float((centers[:, 2] - radii).min())
        if c >= margin - 1e-4:
            return body_q, err, c
        t[2] += margin - c
    bench.joint_q[:] = q_prev
    body_q = bench._fk()
    centers, radii = sphere_state_for(specs, body_q)
    return body_q, float("inf"), float((centers[:, 2] - radii).min())


# wrist-local palm-normal axes, measured from the collider geometry at the
# calibrated home pose (fingers x lateral, oriented toward the thumb side)
PALM_AXIS = {
    "l": np.array([-0.2405, -0.9113, -0.2152]),
    "r": np.array([-0.2405, +0.9113, -0.2152]),
}
PALM_UP = np.array([0.0, 0.0, 1.0])


class KeyframeTrajectory:
    """User-authored keyframes (pose_studio.py), joint-space interpolated.

    Per 30 Hz frame: left_q/right_q (6 arm joints each), fingers/thumb
    fractions per hand, and a per-hand `grip` flag (drives sticky pads +
    grip bridge on replay). The first keyframe is held for `settle_s`, each
    subsequent one is reached over its recorded duration (smoothstep), the
    last is held for `hold_s`. A segment's grip flags switch to the target
    keyframe's at its end.
    """

    def __init__(self, path, fps=30.0, settle_s=0.5, hold_s=1.0):
        d = np.load(path)
        K = len(d["dur"])
        if K < 2:
            raise ValueError("need at least 2 keyframes")
        self.fps = fps
        lq, rq, fi, th, gr = [], [], [], [], []

        def emit(q_l, q_r, f2, t2, g2, n):
            for _ in range(max(1, n)):
                lq.append(q_l); rq.append(q_r)
                fi.append(f2); th.append(t2); gr.append(g2)

        emit(d["left_q"][0], d["right_q"][0], d["fingers"][0], d["thumb"][0],
             d["grip"][0], int(settle_s * fps))
        for i in range(1, K):
            n = max(1, int(round(float(d["dur"][i]) * fps)))
            for s in range(n):
                a = (s + 1) / n
                a = a * a * (3.0 - 2.0 * a)
                lq.append(d["left_q"][i - 1] + a * (d["left_q"][i] - d["left_q"][i - 1]))
                rq.append(d["right_q"][i - 1] + a * (d["right_q"][i] - d["right_q"][i - 1]))
                fi.append(d["fingers"][i - 1] + a * (d["fingers"][i] - d["fingers"][i - 1]))
                th.append(d["thumb"][i - 1] + a * (d["thumb"][i] - d["thumb"][i - 1]))
                gr.append(d["grip"][i] if s == n - 1 else d["grip"][i - 1])
        emit(d["left_q"][-1], d["right_q"][-1], d["fingers"][-1],
             d["thumb"][-1], d["grip"][-1], int(hold_s * fps))
        self.left_q = np.asarray(lq)
        self.right_q = np.asarray(rq)
        self.fingers = np.asarray(fi)
        self.thumb = np.asarray(th)
        self.grip = np.asarray(gr).astype(bool)
        self.n_frames = len(lq)


class TopGraspTearTrajectory:
    """Dual-arm top grasp tear (sketch-specified sequence):

    1. open hands hover ON TOP of the loaf ends, fingers reaching across
    2. descend; the open fingers pass beside the near face to the table
    3. curl the fingers until they touch the table / the bread
    4. close the thumb until the bread is held (transverse clamp)
    5. lift, pull apart -> the bread breaks.

    Hands sit slightly on the near side (y offset) so the descending
    fingers straddle the loaf instead of stabbing its top. Channels per
    frame: TCP target, finger command, thumb command, phase name.
    """

    def __init__(self, x_end, y, z_low, fps=30.0, hover_z=0.17, via_z=0.22,
                 lift_dz=0.05, pull_dx=0.12, pull_speed=0.15,
                 arch=0.25, thumb_rest=0.3, y_stagger=0.0,
                 start_l=None, start_r=None):
        self.fps = fps
        pull_t = pull_dx / pull_speed
        # (duration, phase, |x|, z, fingers cmd, thumb cmd at end)
        # "start"+"via" are the lead-in: the driver joint-lerps into the
        # piano basin there instead of tracking these targets
        plan = [
            (0.5, "start", None, None, arch, thumb_rest),
            (1.5, "via",    x_end, via_z, arch, thumb_rest),
            (1.0, "hover",  x_end, hover_z, arch, thumb_rest),
            (1.2, "descend", x_end, z_low, arch, thumb_rest),
            (0.8, "curl",   x_end, z_low, 0.85, thumb_rest),
            (0.6, "clamp",  x_end, z_low, 0.85, 1.0),
            (0.3, "settle", x_end, z_low, 0.85, 1.0),
            (0.8, "lift",   x_end, z_low + lift_dz, 0.85, 1.0),
            (pull_t, "pull", x_end + pull_dx, z_low + lift_dz, 0.85, 1.0),
            (1.0, "hold",   x_end + pull_dx, z_low + lift_dz, 0.85, 1.0),
        ]
        tl, tr, fi, th, ph = [], [], [], [], []
        pl = np.asarray(start_l, dtype=np.float64)
        pr = np.asarray(start_r, dtype=np.float64)
        pf = pt = 0.0
        for dur, phase, xa, z, f_end, t_end in plan:
            # y_stagger separates the two hands' interleaving fingers
            gl = pl if xa is None else np.array([-xa, y - y_stagger, z])
            gr = pr if xa is None else np.array([+xa, y + y_stagger, z])
            n = max(1, int(round(dur * fps)))
            for i in range(n):
                a = (i + 1) / n
                a = a * a * (3.0 - 2.0 * a)
                tl.append(pl + a * (gl - pl))
                tr.append(pr + a * (gr - pr))
                fi.append(pf + a * (f_end - pf))
                th.append(pt + a * (t_end - pt))
                ph.append(phase)
            pl, pr, pf, pt = gl, gr, f_end, t_end
        self.targets_l = np.asarray(tl)
        self.targets_r = np.asarray(tr)
        self.fingers = np.asarray(fi)
        self.thumb = np.asarray(th)
        self.phase = ph
        self.n_frames = len(th)


class ScoopTearTrajectory:
    """Dual-arm scoop-and-clamp tear (the human strategy):

    flat palms low outside the loaf ends -> slide inboard so the flat
    fingers wedge UNDER the end caps (the bread rests on the palms) ->
    thumbs clamp down on top -> lift -> pull apart.

    Palm-up orientation comes from tracking PALM_AXIS -> PALM_UP (fingers
    then naturally point outboard, which is what makes the axial scoop
    work). Per-frame: TCP targets (rig meters), thumb command, and the
    phase name. The four fingers stay flat throughout.
    """

    def __init__(self, x_stage, x_scoop, z_low, fps=30.0, via_z=0.20,
                 lift_dz=0.05, pull_dx=0.12, pull_speed=0.15,
                 y=0.30, start_l=None, start_r=None):
        self.fps = fps
        pull_t = pull_dx / pull_speed
        # (duration, phase, |x| target, z target, thumb command at end)
        plan = [
            (0.5, "start", None, None, 0.0),
            (1.5, "via",   x_stage + 0.03, via_z, 0.0),
            (1.2, "stage", x_stage, z_low, 0.0),
            (1.4, "scoop", x_scoop, z_low, 0.0),
            (0.6, "clamp", x_scoop, z_low, 1.0),
            (0.3, "settle", x_scoop, z_low, 1.0),
            (0.8, "lift",  x_scoop, z_low + lift_dz, 1.0),
            (pull_t, "pull", x_scoop + pull_dx, z_low + lift_dz, 1.0),
            (1.0, "hold",  x_scoop + pull_dx, z_low + lift_dz, 1.0),
        ]
        tl, tr, th, ph = [], [], [], []
        pl = np.asarray(start_l, dtype=np.float64)
        pr = np.asarray(start_r, dtype=np.float64)
        pt = 0.0
        for dur, phase, xa, z, t_end in plan:
            gl = pl if xa is None else np.array([-xa, y, z])
            gr = pr if xa is None else np.array([+xa, y, z])
            n = max(1, int(round(dur * fps)))
            for i in range(n):
                a = (i + 1) / n
                a = a * a * (3.0 - 2.0 * a)
                tl.append(pl + a * (gl - pl))
                tr.append(pr + a * (gr - pr))
                th.append(pt + a * (t_end - pt))
                ph.append(phase)
            pl, pr, pt = gl, gr, t_end
        self.targets_l = np.asarray(tl)
        self.targets_r = np.asarray(tr)
        self.thumb = np.asarray(th)
        self.phase = ph
        self.n_frames = len(th)


class ScriptedTearTrajectory:
    """Dual-arm mirrored tear: hover -> descend -> close -> lift -> pull -> hold.

    Per-frame TCP targets for both arms (rig meters) + finger fraction.
    `end_l`/`end_r` are the loaf-end grasp points; the pull moves each hand
    outward along +-x by `pull_dx` at `pull_speed`.
    """

    def __init__(self, end_l, end_r, grasp_z, fps=30.0, hover_z=0.16,
                 via_z=0.26, lift_dz=0.03, pull_dx=0.12, pull_speed=0.15,
                 close_time_s=0.6, start_l=None, start_r=None):
        self.fps = fps
        eL = np.asarray(end_l, dtype=np.float64)
        eR = np.asarray(end_r, dtype=np.float64)
        pull_t = pull_dx / pull_speed

        def wp_(e, sign, phase):
            if phase == "via":
                return [e[0], e[1], via_z]
            if phase == "hover":
                return [e[0], e[1], hover_z]
            if phase == "grasp":
                return [e[0], e[1], grasp_z]
            if phase == "lift":
                return [e[0], e[1], grasp_z + lift_dz]
            if phase == "pull":
                return [e[0] + sign * pull_dx, e[1], grasp_z + lift_dz]
            raise KeyError(phase)

        # (duration, phase-key at end, finger frac at end)
        plan = [
            (1.5, "via",   0.0),
            (1.0, "hover", 0.0),
            (1.2, "grasp", 0.0),
            (close_time_s, "grasp", 1.0),
            (0.4, "grasp", 1.0),   # settle
            (0.8, "lift",  1.0),
            (pull_t, "pull", 1.0),
            (1.0, "pull",  1.0),   # hold
        ]
        if start_l is not None:
            plan = [(0.5, "start", 0.0)] + plan

        tl, tr, fr = [], [], []
        pl = np.asarray(start_l if start_l is not None else wp_(eL, -1, "via"),
                        dtype=np.float64)
        pr = np.asarray(start_r if start_r is not None else wp_(eR, +1, "via"),
                        dtype=np.float64)
        pf = 0.0
        for dur, phase, f_end in plan:
            gl = pl if phase == "start" else np.asarray(wp_(eL, -1.0, phase))
            gr = pr if phase == "start" else np.asarray(wp_(eR, +1.0, phase))
            n = max(1, int(round(dur * fps)))
            for i in range(n):
                a = (i + 1) / n
                a = a * a * (3.0 - 2.0 * a)  # smoothstep
                tl.append(pl + a * (gl - pl))
                tr.append(pr + a * (gr - pr))
                fr.append(pf + a * (f_end - pf))
            pl, pr, pf = gl, gr, f_end
        self.targets_l = np.asarray(tl)
        self.targets_r = np.asarray(tr)
        self.fracs = np.asarray(fr)
        self.n_frames = len(fr)
        g = np.flatnonzero(self.fracs > 0.6)
        self.grasp_frame = int(g[0]) if len(g) else self.n_frames


class ScriptedGraspTrajectory:
    """Top-down grasp: hover -> descend -> close -> lift -> hold.

    Interface: fps, n_frames, contact,
    src_index, pinch_T, finger_frac, pinch_point_rig) so run_coupling can use
    either. Targets are TCP (finger-pad centroid) positions in rig meters.
    """

    def __init__(self, block_center, block_half_z, fps=30.0,
                 hover_z=0.16, lift_z=0.20, close_time_s=0.5, start_point=None):
        self.fps = fps
        c = np.asarray(block_center, dtype=np.float64)
        grasp_z = max(c[2] - 0.008, 0.012)  # pads engulf the core, floor above table
        phases = []
        if start_point is not None:
            phases.append((0.8, list(start_point), 0.0))      # hold home
            phases.append((1.4, [c[0], c[1], hover_z], 0.0))  # home -> hover
        phases += [
            (1.0, [c[0], c[1], hover_z], 0.0),   # hover
            (1.2, [c[0], c[1], grasp_z], 0.0),   # descend
            (close_time_s, [c[0], c[1], grasp_z], 1.0),  # close
            (0.4, [c[0], c[1], grasp_z], 1.0),   # settle
            (1.6, [c[0], c[1], lift_z], 1.0),    # lift
            (1.2, [c[0], c[1], lift_z], 1.0),    # hold
        ]
        targets, fracs = [], []
        prev_p = np.array(phases[0][1], dtype=np.float64)
        prev_f = 0.0
        for dur, p, f in phases:
            n = max(1, int(round(dur * fps)))
            p = np.asarray(p, dtype=np.float64)
            for i in range(n):
                a = (i + 1) / n
                a = a * a * (3.0 - 2.0 * a)  # smoothstep
                targets.append(prev_p + a * (p - prev_p))
                fracs.append(prev_f + a * (f - prev_f))
            prev_p, prev_f = p, f
        self.targets = np.asarray(targets)
        self.fracs = np.asarray(fracs)
        self.n_frames = len(targets)
        self.contact = self.fracs > 0.6
        self.src_index = np.arange(self.n_frames)
        # pinch_T-compatible: identity rotation, target positions
        self.pinch_T = np.tile(np.eye(4), (self.n_frames, 1, 1))
        self.pinch_T[:, :3, 3] = self.targets
        g = np.flatnonzero(self.contact)
        self.grasp_frame = int(g[0]) if len(g) else self.n_frames
        self.release_frame = self.n_frames
        self._block_center = c

    def finger_frac(self, frame: int) -> float:
        return float(self.fracs[min(frame, self.n_frames - 1)])

    def pinch_point_rig(self) -> np.ndarray:
        return self._block_center.copy()
