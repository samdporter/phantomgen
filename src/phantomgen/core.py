import numpy as np
import argparse

# --------------------------- Geometry helpers ---------------------------

def _world_coords(shape, voxel_mm, center_mm):
    """Return world coordinates (mm) for voxel centers with a given center shift."""
    z, y, x = [np.arange(n, dtype=float) for n in shape]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    sz, sy, sx = voxel_mm
    cz, cy, cx = center_mm
    Z = (Z - (shape[0]-1)/2) * sz - cz
    Y = (Y - (shape[1]-1)/2) * sy - cy
    X = (X - (shape[2]-1)/2) * sx - cx
    return Z, Y, X

def add_box(volume, voxel_mm, size_mm, center_mm, value):
    """Solid axis-aligned box centered at center_mm."""
    Z, Y, X = _world_coords(volume.shape, voxel_mm, center_mm)
    dz, dy, dx = [s/2 for s in size_mm]
    mask = (np.abs(Z) <= dz) & (np.abs(Y) <= dy) & (np.abs(X) <= dx)
    volume[mask] = value

def add_cylinder(volume, voxel_mm, radius_mm, height_mm, deg_range, center_mm, value):
    """Z-axis cylinder (optionally angularly clipped). deg_range=(start,end) in degrees or None."""
    Z, Y, X = _world_coords(volume.shape, voxel_mm, center_mm)
    mask = (np.abs(Z) <= height_mm/2) & ((X**2 + Y**2) <= radius_mm**2)
    if deg_range is not None:
        s, e = (deg_range[0] % 360.0, deg_range[1] % 360.0)
        span = (e - s) % 360.0
        if span and abs(span - 360.0) > 1e-9:
            theta = np.degrees(np.arctan2(Y, X)) % 360.0
            mask &= (theta >= s) & (theta <= e) if s <= e else ((theta >= s) | (theta <= e))
    volume[mask] = value

def add_sphere(volume, voxel_mm, radius_mm, center_mm, value):
    Z, Y, X = _world_coords(volume.shape, voxel_mm, center_mm)
    mask = (X**2 + Y**2 + Z**2) <= radius_mm**2
    volume[mask] = value

# --------------------------- Phantom builder ---------------------------

def create_nema(matrix_size=(256, 256, 256),
                voxel_size_mm=(2.0, 2.0, 2.0),
                nema_dict=None,
                center_offset_mm=None):
    """
    Build NEMA IQ-style phantom with optional global offset (mm) (cz, cy, cx).
    Returns (activity_volume, ctac_volume).
    """
    # Defaults (kept equivalent to your original)
    earl_nema_dict = {
        "activity_concentration_background": 0.05,  # MBq/ml
        "fill_mu_value": 0.096,                     # cm^-1
        "perspex_mu_value": 0.12,
        "lung_insert": {"include": True, "lung_mu_value": 0.029},
        "sphere_dict": {
            "ring_R": 57, "ring_z": -37,
            "spheres": {
                "diametre_mm":     [10, 13, 17, 22, 28, 37],
                "angle_loc":       [30, 90, 150, 210, 270, 330],
                "act_conc_MBq_ml": [0.00, 0.00, 0.04, 0.04, 0.04, 0.04]
            }
        },
        "center_offset_mm": (0.0, 0.0, 0.0),
    }
    pet_nema_dict = {
        **earl_nema_dict,
        "activity_concentration_background": 0.00,  # PET style blank background (as in your file)
    }

    cfg = pet_nema_dict if (nema_dict == "pet") else (nema_dict or earl_nema_dict)
    # allow CLI or dict-provided offset; CLI wins if provided
    offset = tuple(center_offset_mm) if center_offset_mm is not None else tuple(cfg.get("center_offset_mm", (0.0, 0.0, 0.0)))
    cz, cy, cx = offset

    # Convenient local alias
    def with_off(c):  # (cz, cy, cx) + local center (z,y,x)
        return (c[0] + cz, c[1] + cy, c[2] + cx)

    # Allocate
    act_vol = np.zeros(matrix_size, dtype=np.float32)
    ctac_vol = np.zeros(matrix_size, dtype=np.float32)

    # Constants
    back_MBq = float(cfg["activity_concentration_background"])
    # Convert MBq/ml -> per-voxel: voxel volume in ml = (mm^3)/1000
    vox_ml = np.prod(voxel_size_mm) / 1000.0
    back_MBq_per_vox = back_MBq * vox_ml

    fill_mu_value = float(cfg["fill_mu_value"])
    perspex_mu_value = float(cfg.get("perspex_mu_value", 0.12))
    lung_insert = cfg.get("lung_insert", {}).get("include", False)
    lung_mu_value = float(cfg.get("lung_insert", {}).get("lung_mu_value", 0.029))

    # ---------------- Tank structure ----------------
    # Outer shell/fill + two side cylinders + optional lung
    tanks = [
        dict(r=150, h=220, deg=(180, 360), c=(0, 35, 0),    mu="perspex"),
        dict(r=147, h=214, deg=(180, 360), c=(0, 35, 0),    mu="fill"),
        dict(r=75,  h=220, deg=(90, 180),  c=(0, 35, -75),  mu="perspex"),
        dict(r=72,  h=214, deg=(90, 180),  c=(0, 35, -75),  mu="fill"),
        dict(r=75,  h=220, deg=(0, 90),    c=(0, 35, 75),   mu="perspex"),
        dict(r=72,  h=214, deg=(0, 90),    c=(0, 35, 75),   mu="fill"),
    ]
    if lung_insert:
        tanks.append(dict(r=25, h=214, deg=None, c=(0, 0, 0), mu="lung"))

    # One small connector box (perspex)
    add_box(ctac_vol, voxel_size_mm, size_mm=(220, 75, 150), center_mm=with_off((0, 72.5, 0)), value=perspex_mu_value)

    # Paint cylinders
    for t in tanks:
        center = with_off(t["c"])
        if t["mu"] == "perspex":
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], center, perspex_mu_value)
        elif t["mu"] == "fill":
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], center, fill_mu_value)
            add_cylinder(act_vol,  voxel_size_mm, t["r"], t["h"], t["deg"], center, back_MBq_per_vox)
        else:  # lung
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], center, lung_mu_value)
            add_cylinder(act_vol,  voxel_size_mm, t["r"], t["h"], t["deg"], center, 0.0)

    # ---------------- Hot spheres ----------------
    sdict = cfg["sphere_dict"]["spheres"]
    ring_R = float(cfg["sphere_dict"]["ring_R"])
    ring_z = float(cfg["sphere_dict"]["ring_z"])

    for d_mm, angle_deg, conc in zip(sdict["diametre_mm"], sdict["angle_loc"], sdict["act_conc_MBq_ml"]):
        r_shell = (d_mm + 2) / 2.0   # 1 mm wall thickness
        r_inner = d_mm / 2.0
        theta = np.deg2rad(angle_deg)
        cy_s = ring_R * np.sin(theta)
        cx_s = ring_R * np.cos(theta)
        c_s = (ring_z, cy_s + 35, cx_s)  # 35 mm lateral offset for ring center (as in original)

        # Perspex shell on CT
        add_sphere(ctac_vol, voxel_size_mm, r_shell, with_off(c_s), perspex_mu_value)

        # Interior: fill (CT) + activity (ACT)
        add_sphere(ctac_vol, voxel_size_mm, r_inner, with_off(c_s), fill_mu_value)
        add_sphere(act_vol,  voxel_size_mm, r_inner, with_off(c_s), conc * vox_ml)

    return act_vol, ctac_vol

# --------------------------- CLI ---------------------------

def _build_parser():
    p = argparse.ArgumentParser(description="Build NEMA-like phantom and save ACT/CTAC volumes (.npy).")
    p.add_argument("--preset", choices=["earl", "pet"], default="earl", help="Parameter preset")
    p.add_argument("--z", type=int, default=256)
    p.add_argument("--y", type=int, default=256)
    p.add_argument("--x", type=int, default=256)
    p.add_argument("--voxel", type=float, nargs=3, default=[2.0, 2.0, 2.0], metavar=("sz","sy","sx"))
    p.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("cz","cy","cx"),
                   help="Global center offset (mm)")
    p.add_argument("--out-act", default="activity.npy")
    p.add_argument("--out-ct",  default="ctac.npy")
    return p

def main():
    args = _build_parser().parse_args()
    matrix_size = (args.z, args.y, args.x)
    voxel_mm = tuple(args.voxel)

    # Select preset dict but keep same keys
    if args.preset == "pet":
        preset = "pet"  # sentinel handled in create_nema to switch defaults
    else:
        preset = None   # use EARL defaults

    act, ct = create_nema(
        matrix_size=matrix_size,
        voxel_size_mm=voxel_mm,
        nema_dict=preset,
        center_offset_mm=tuple(args.offset),
    )
    np.save(args.out_act, act)
    np.save(args.out_ct, ct)
    print(f"Saved:\n  {args.out_act}  (shape {act.shape}, dtype {act.dtype})\n  {args.out_ct}  (shape {ct.shape}, dtype {ct.dtype})")

if __name__ == "__main__":
    main()
