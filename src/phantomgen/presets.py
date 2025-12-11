"""NEMA phantom preset configurations."""

earl_nema_dict = {
    "mu_values": {
        "perspex_mu_value": 0.15,
        "fill_mu_value": 0.14,
        "lung_mu_value": 0.043
    },
    "activity_concentration_background": 0.0,
    "include_lung_insert": False,
    "center_offset_mm": (0.0, 0.0, 0.0),
    "sphere_dict": {
        "ring_R": 57,
        "ring_x": 0,
        "ring_y": 0,
        "ring_z": -37,
        "spheres": {
            "diametre_mm": [13, 17, 22, 28, 37, 60],
            "angle_loc": [270, 150, 30, 90, 330, 210],
            "act_conc_MBq_ml": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    }
}

pet_nema_dict = {
    "mu_values": {
        "perspex_mu_value": 0.1,
        "fill_mu_value": 0.096,
        "lung_mu_value": 0.029
    },
    "activity_concentration_background": 0.05,
    "include_lung_insert": True,
    "center_offset_mm": (0.0, 0.0, 0.0),
    "sphere_dict": {
        "ring_R": 57,
        "ring_x": 0,
        "ring_y": 0,
        "ring_z": -37,
        "spheres": {
            "diametre_mm": [10, 13, 17, 22, 28, 37],
            "angle_loc": [30, 90, 150, 210, 270, 330],
            "act_conc_MBq_ml": [0.00, 0.00, 0.4, 0.4, 0.4, 0.4],
        }
    }
}
