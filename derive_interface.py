"""
Derivation helper for interface conductance using the Yan et al. two-layer model,
with custom channel height H_t and substrate thickness H_b, and optional Nusselt
correlation override.

Note: The current solver applies the full interface conductance (h_int * area),
without dividing by (2*H_t) or (2*H_b). The h_int computed here is the value that
the solver will use directly when h_scale=1.

Usage:
    python derive_interface.py
Edit the constants at the bottom to match your case.
"""

from dataclasses import dataclass


@dataclass
class Geometry:
    H_t: float  # half-height of thermal-fluid layer (m)
    H_b: float  # half-height of substrate (m)


@dataclass
class Properties:
    k_f: float  # fluid conductivity (W/m-K)
    k_s: float  # solid conductivity (W/m-K)
    Nu_override: float = None  # if provided, use Nu*k_f/H_t instead of 35/26*k_f/H_t


def compute_ht(props: Properties, geom: Geometry) -> float:
    """
    Fluid-side interface coefficient.
    Paper form: h_t = 35*k_f / (26*H_t)
    If Nu_override is provided: h_t = Nu*k_f / H_t
    """
    if props.Nu_override is not None:
        return props.Nu_override * props.k_f / geom.H_t
    return 35.0 * props.k_f / (26.0 * geom.H_t)


def compute_hb(props: Properties, geom: Geometry) -> float:
    """Substrate-side coefficient: h_b = k_s / H_b"""
    return props.k_s / geom.H_b


def series_h(h_t: float, h_b: float) -> float:
    """Interface conductance in series: h = (h_t*h_b)/(h_t + h_b)"""
    return (h_t * h_b) / (h_t + h_b + 1e-20)


def required_scale(q_flux: float, desired_deltaT: float, h_unscaled: float) -> float:
    """
    If you want q_flux to drive desired_deltaT across the interface,
    you need h_target = q_flux / desired_deltaT.
    The scale factor on h_unscaled is h_target / h_unscaled.
    """
    h_target = q_flux / desired_deltaT
    return h_target / h_unscaled


def main():
    # Edit these for your case
    geom = Geometry(H_t=0.004, H_b=0.0005)  # 4 mm half height, 0.5 mm substrate half
    props = Properties(k_f=0.6, k_s=400.0, Nu_override=None)
    q_flux = 1.0e6  # W/m^2
    desired_deltaT = 10.0  # K target interface jump

    h_t = compute_ht(props, geom)
    h_b = compute_hb(props, geom)
    h_int = series_h(h_t, h_b)
    scale_needed = required_scale(q_flux, desired_deltaT, h_int)

    print("Geometry:")
    print(f"  H_t = {geom.H_t} m")
    print(f"  H_b = {geom.H_b} m")
    print("Properties:")
    print(f"  k_f = {props.k_f} W/m-K")
    print(f"  k_s = {props.k_s} W/m-K")
    print(f"  Nu_override = {props.Nu_override}")
    print("\nDerived (paper form unless Nu_override set):")
    print(f"  h_t = {h_t:.2f} W/m^2-K")
    print(f"  h_b = {h_b:.2f} W/m^2-K")
    print(f"  h_int (series) = {h_int:.2f} W/m^2-K")
    print("\nFor q\" = {:.2e} W/m^2 and desired ΔT = {:.2f} K:".format(q_flux, desired_deltaT))
    print(f"  Required h_target = {q_flux/desired_deltaT:.2f} W/m^2-K")
    print(f"  Scale factor on h_int needed ≈ {scale_needed:.1f}x")


if __name__ == "__main__":
    main()

