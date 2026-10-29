"""Export comparable compiled model parameters for an isolated backend audit.

Run once in each explicit environment on the server; compare the resulting
files before claiming physics-model equivalence. This does not advance physics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


FIELDS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "jnt_pos", "jnt_axis", "jnt_range", "jnt_stiffness", "dof_armature", "dof_damping",
    "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
    "actuator_ctrlrange", "geom_pos", "geom_quat", "geom_size", "geom_friction",
    "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "geom_condim",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    arrays = {name: np.array(getattr(model, name), copy=True) for name in FIELDS}
    np.savez(args.output, **arrays)
    meta = {"mujoco": mujoco.__version__, "scene": str(args.scene.resolve()),
            "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
            "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "nbody": model.nbody, "timestep": model.opt.timestep,
            "solver": int(model.opt.solver), "integrator": int(model.opt.integrator),
            "iterations": model.opt.iterations, "ls_iterations": model.opt.ls_iterations}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
