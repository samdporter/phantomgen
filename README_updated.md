# phantomgen

**phantomgen** is a Python tool for generating 3D numerical NEMA‐like image quality phantoms used in SPECT and PET imaging research.  
It creates voxelized *activity* and *CT attenuation* maps suitable for simulation and reconstruction studies.

---

## ✨ Features

- Generates a standard **NEMA IQ phantom** (SPECT or PET‐style).
- Supports **configurable matrix size** and **voxel dimensions**.
- Outputs both **activity** and **CT attenuation** volumes (`.npy`).
- Allows **global spatial offset** in millimeters via the new `--offset` argument.
- Easily integrated into simulation or reconstruction pipelines.

---

## 🧠 Phantom Description

The phantom consists of:
- A main cylindrical body and side compartments.
- Six spherical inserts (10–37 mm diameters).
- Optional lung insert (low attenuation region).
- Perspex walls and connecting box geometry.

Two default presets are available:
- **`earl`** – SPECT/Hybrid phantom with nonzero background activity.
- **`pet`** – PET phantom with zero background activity.

---

## ⚙️ Command-Line Usage

```bash
python core.py [options]
```

### Basic example
```bash
python core.py --preset earl --z 256 --y 256 --x 256                --voxel 2 2 2                --out-act activity.npy                --out-ct ctac.npy
```

### New: Applying a global offset
You can now shift the entire phantom in world coordinates by specifying an offset in **millimeters** (Z, Y, X order):

```bash
python core.py --preset earl                --offset 10.0 5.0 -5.0
```

This applies a **+10 mm shift along Z**, **+5 mm along Y**, and **−5 mm along X** before geometry creation.  
The offset is applied uniformly to all primitives (tanks, spheres, boxes, etc.).

> 💡 Use this if you need to simulate a phantom that is slightly off-center within a larger FOV.

---

## 🧩 Python API

```python
from core import create_nema

act_vol, ctac_vol = create_nema(
    matrix_size=(256, 256, 256),
    voxel_size_mm=(2.0, 2.0, 2.0),
    nema_dict="earl",               # or "pet" or a custom dict
    center_offset_mm=(10.0, 5.0, -5.0)  # optional global offset
)
```

Both returned volumes are NumPy arrays with the same shape as the requested matrix.

---

## 📦 Outputs

| File | Description | Type |
|------|--------------|------|
| `activity.npy` | Activity map (MBq per voxel) | `float32` |
| `ctac.npy` | Attenuation map (cm⁻¹) | `float32` |

You can load them as:
```python
import numpy as np
act = np.load("activity.npy")
ct  = np.load("ctac.npy")
```

---

## 🧾 Default presets

| Parameter | EARL preset | PET preset |
|------------|--------------|-------------|
| Background activity (MBq/ml) | 0.05 | 0.00 |
| Fill μ (cm⁻¹) | 0.096 | 0.096 |
| Perspex μ (cm⁻¹) | 0.12 | 0.12 |
| Lung μ (cm⁻¹) | 0.029 | 0.029 |

---

## 🧠 Notes

- The offset is **optional**; if omitted, the phantom is centered at the volume origin.
- Units are always in **millimeters** for geometry and **cm⁻¹** for attenuation.
- The Z, Y, X order matches the NumPy array indexing convention used internally.

---

## 📚 Citation

If you use **phantomgen** in research, please cite it as:

> *Varzakis E.*, *Porter S.*, *et al.* “Numerical generation of NEMA IQ-style phantoms for hybrid molecular imaging simulations.”  
> (Institute of Nuclear Medicine, UCL, 2025).

---

## 🧩 License

This project is distributed under the MIT License.

---

**Authors:**  
- E. Varzakis (UCL Institute of Nuclear Medicine)  
- S. Porter — *Phantom centering and offset implementation*  
