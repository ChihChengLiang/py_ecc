from typing import (
    Tuple,
)
from py_ecc.fields import (
    optimized_bls12_381_FQ2 as FQ2,
)
from py_ecc.typing import (
    Optimized_Point3D,
)

from .constants import (
    ISO_3_A,
    ISO_3_B,
    ISO_3_Z,
    P_MINUS_9_DIV_16,
    ETAS,
    ISO_3_MAP_COEFFICIENTS,
    POSITIVE_EIGTH_ROOTS_OF_UNITY,
)


# Optimized SWU Map - FQ2 to G2': y^2 = x^3 + 240i * x + 1012 + 1012i
# Found in Section 4 of https://eprint.iacr.org/2019/403
def optimized_swu_G2(t: FQ2) -> Tuple[FQ2, FQ2, FQ2]:
    t2 = t ** 2
    iso_3_z_t2 = ISO_3_Z * t2
    zt2_z2t4 = iso_3_z_t2 + iso_3_z_t2 ** 2
    denominator = -(ISO_3_A * zt2_z2t4)  # -a(Z * t^2 + Z^2 * t^4)
    numerator = ISO_3_B * (zt2_z2t4 + FQ2.one())  # b(Z * t^2 + Z^2 * t^4 + 1)

    # Exceptional case
    if denominator == FQ2.zero():
        denominator = ISO_3_Z * ISO_3_A

    # v = D^3
    v = denominator ** 3
    # u = N^3 + a * N * D^2 + b* D^3
    u = (numerator ** 3) + (ISO_3_A * numerator * (denominator ** 2)) + (ISO_3_B * v)

    # Attempt y = sqrt(u / v)
    (success, alpha) = sqrt_division_FQ2(u, v)
    if success:
        y = alpha if t.sgn0_be() == alpha.sgn0_be() else -alpha
        return (numerator, y * denominator, denominator)
    else:
        # Handle case where (u / v) is not square
        # sqrt_candidate(x1) = sqrt_candidate(x0) * t^3
        alpha_t3 = alpha * t ** 3

        # u(x1) = Z^3 * t^6 * u(x0)
        u_x1 = iso_3_z_t2 ** 3 * u
        success_2 = False
        for eta in ETAS:
            # Valid solution if (eta * sqrt_candidate(x1)) ** 2 * v - u == 0
            eta_alpha_t3 = eta * alpha_t3
            determinant = eta_alpha_t3 ** 2 * v - u_x1
            if determinant == FQ2.zero() and not success_2:
                y = eta_alpha_t3
                success_2 = True
        else:
            if not success and not success_2:
                # Unreachable
                raise Exception("Hash to Curve - Optimized SWU failure")
        y = y if t.sgn0_be() == y.sgn0_be() else -y
        return (numerator * iso_3_z_t2, y * denominator, denominator)


# Square Root Division
# Return: uv^7 * (uv^15)^((p^2 - 9) / 16) * root of unity
# If valid square root is found return true, else false
def sqrt_division_FQ2(u: FQ2, v: FQ2) -> Tuple[bool, FQ2]:
    u_v7 = u * v ** 7
    u_v15 = u_v7 * v ** 8

    # gamma =  uv^7 * (uv^15)^((p^2 - 9) / 16)
    gamma = u_v7 * u_v15 ** P_MINUS_9_DIV_16

    # Verify there is a valid root
    is_valid_root = False
    result = gamma
    for root in POSITIVE_EIGTH_ROOTS_OF_UNITY:
        # Valid if (root * gamma)^2 * v - u == 0
        sqrt_candidate = (root * gamma)
        determinant = sqrt_candidate ** 2 * v - u
        if determinant == FQ2.zero() and not is_valid_root:
            is_valid_root = True
            result = sqrt_candidate

    return (is_valid_root, result)


# Optimal Map from 3-Isogenous Curve to G2
def iso_map_G2(x: FQ2, y: FQ2, z: FQ2) -> Optimized_Point3D[FQ2]:
    # x-numerator, x-denominator, y-numerator, y-denominator
    mapped_values = [FQ2.zero(), FQ2.zero(), FQ2.zero(), FQ2.zero()]
    z_powers = [z, z ** 2, z ** 3]

    # Horner Polynomial Evaluation
    for (i, k_i) in enumerate(ISO_3_MAP_COEFFICIENTS):
        mapped_values[i] = k_i[-1:][0]
        for (j, k_i_j) in enumerate(reversed(k_i[:-1])):
            mapped_values[i] = mapped_values[i] * x + z_powers[j] * k_i_j

    mapped_values[2] = mapped_values[2] * y  # y-numerator * y
    mapped_values[3] = mapped_values[3] * z  # y-denominator * z

    z_G2 = mapped_values[1] * mapped_values[3]  # x-denominator * y-denominator
    x_G2 = mapped_values[0] * mapped_values[3]  # x-numerator * y-denominator
    y_G2 = mapped_values[1] * mapped_values[2]  # y-numerator * x-denominator

    return (x_G2, y_G2, z_G2)
