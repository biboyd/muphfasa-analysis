"""yt loading helper: muphfasa plotfiles are in MKS, but yt's AMReX frontend assumes cgs."""
import yt

MKS_CODE_UNITS = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "velocity_unit": (1.0, "m/s"),
}

# density is a "known" boxlib field that yt tags as g/cm**3; vx/vy/vz are unknown and
# would otherwise be dimensionless.
FIELD_UNITS = {
    ("boxlib", "density"): "kg/m**3",
    ("boxlib", "vx"): "m/s",
    ("boxlib", "vy"): "m/s",
    ("boxlib", "vz"): "m/s",
}


def load_mks(filename):
    """yt.load a muphfasa plotfile with code units and field units set to MKS."""
    ds = yt.load(filename, unit_system="mks", units_override=MKS_CODE_UNITS)
    ds.field_units.update(FIELD_UNITS)
    return ds
