"""Local stand-in for roborender's `roborender_sim.robots` (private, lives on
the original author's machine and is not on PyPI).

Provides `add_yam_sharpa(scene, xform, side)` with the interface surface
robot_scene.py actually uses:

- parses assets/yam_sharpa/yam_sharpa_{side}.urdf into a sub ModelBuilder at
  scale=100 (centimeters) and merges it into `scene` with an `a{b0}` label
  prefix (robot_scene recovers the arm subtree from that prefix);
- geometry (<visual>/<collision>) is stripped before parsing: the mesh files
  are not shipped and the MPM coupling only uses analytic
  hand spheres placed on the bodies, never Newton collision shapes;
- returns an ArmHandle with dof_start / arm_dof_count / dof_count,
  finger_open_q / finger_close_q (open = URDF zero, close = curled flexion
  posture; robot_scene._anchor_pad_spheres self-calibrates the pad spheres and
  TCP at the closed posture, so exact angles are not critical),
  ee_body (the `{side}_hand_wrist` body), ee_offset (identity; robot_scene
  re-measures it from the closed-pad centroid), and down_y (unit quaternion —
  every ik.track call in this repo runs with rot_weight=0).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from newton import ModelBuilder

# robot_scene.py overrides this to the repository's asset/robot/ directory
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

ARM_DOF = 6


def _strip_geometry(urdf_path: Path) -> str:
    """Return URDF XML with <collision> removed (contact is analytic hand
    spheres in the MPM, never Newton collision) and <visual> kept only when
    the referenced mesh files actually exist (they are used purely for
    rendering in render_mesh.py). Mesh paths are made absolute so the XML
    string can be parsed from anywhere."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    base = urdf_path.parent
    for link in root.iter("link"):
        for el in link.findall("collision"):
            link.remove(el)
        for el in link.findall("visual"):
            keep = True
            for mesh in el.iter("mesh"):
                fn = mesh.get("filename", "")
                p = (base / fn).resolve()
                if p.is_file():
                    mesh.set("filename", str(p))
                else:
                    keep = False
            if not keep:
                link.remove(el)
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def _finger_close_value(joint_name: str, lower: float, upper: float) -> float:
    """Closed-posture angle for one hand joint (radians)."""
    n = joint_name.lower()
    if n.endswith("_aa"):
        return 0.0  # no ab/adduction in the closed posture
    if "thumb_cmc_fe" in n:
        return min(1.2, 0.65 * upper)  # oppose the thumb
    if "pinky_cmc" in n and "fe" not in n:
        return 0.3 * upper  # slight palm curl
    # flexion joints: MCP_FE / PIP / DIP / IP
    return 0.6 * upper


def add_yam_sharpa(scene: ModelBuilder, xform, side: str = "left"):
    urdf_path = Path(ASSETS_DIR) / "yam_sharpa" / f"yam_sharpa_{side}.urdf"
    xml = _strip_geometry(urdf_path)

    sub = ModelBuilder()
    sub.add_urdf(
        xml,
        scale=100.0,  # URDF meters -> scene centimeters
        floating=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
    )

    dof_start = len(scene.joint_q)
    body_offset = len(scene.body_label)
    dof_count = len(sub.joint_q)

    # per-coordinate joint name (revolute joints: 1 coord each; fixed: 0)
    coord_joint = [""] * dof_count
    for j, name in enumerate(sub.joint_label):
        q0 = sub.joint_q_start[j]
        q1 = (sub.joint_q_start[j + 1] if j + 1 < len(sub.joint_label)
              else dof_count)
        for c in range(q0, q1):
            coord_joint[c] = name

    # add_urdf prefixes labels with the robot name ("yam_sharpa_left/...")
    arm_joints = [f"{side}_dof_joint{k}" for k in range(1, ARM_DOF + 1)]
    got = [n.rsplit("/", 1)[-1] for n in coord_joint[:ARM_DOF]]
    assert got == arm_joints, f"expected the 6 arm joints first, got {got}"

    lower = np.asarray(sub.joint_limit_lower, dtype=np.float64)
    upper = np.asarray(sub.joint_limit_upper, dtype=np.float64)

    n_fingers = dof_count - ARM_DOF
    finger_open_q = np.zeros(n_fingers)
    finger_close_q = np.zeros(n_fingers)
    for i in range(n_fingers):
        c = ARM_DOF + i
        finger_close_q[i] = np.clip(
            _finger_close_value(coord_joint[c], lower[c], upper[c]),
            lower[c], upper[c])

    ee_link = f"{side}_hand_wrist"
    shorts = [l.rsplit("/", 1)[-1] for l in sub.body_label]
    ee_local = (shorts.index(ee_link) if ee_link in shorts
                else shorts.index(f"{side}_wrist_mount"))

    scene.add_builder(sub, xform=xform, label_prefix=f"a{body_offset}")

    return SimpleNamespace(
        side=side,
        dof_start=dof_start,
        arm_dof_count=ARM_DOF,
        dof_count=dof_count,
        finger_open_q=finger_open_q,
        finger_close_q=finger_close_q,
        finger_joint_names=[n.rsplit("/", 1)[-1] for n in coord_joint[ARM_DOF:]],
        ee_body=body_offset + ee_local,
        ee_offset=wp.transform_identity(),
        down_y=(0.0, 0.0, 0.0, 1.0),
    )
