from typing import (  # noqa: F401
    Dict,
    Sequence,
    Tuple,
    Union,
)

from eth_utils import (
    big_endian_to_int,
)
from py_ecc.optimized_bls12_381 import (
    FQ,
    FQ2,
    FQP,
    b,
    b2,
    field_modulus as q,
    is_on_curve,
    multiply,
    normalize,
)

from .hash import (
    hash_eth2,
)
from .typing import (
    BLSPubkey,
    BLSSignature,
    Hash32,
)

G2_cofactor = 305502333931268344200999753193121504214466019254188142667664032982267604182971884026507427359259977847832272839041616661285803823378372096355777062779109  # noqa: E501
FQ2_order = q ** 2 - 1
eighth_roots_of_unity = [
    FQ2([1, 1]) ** ((FQ2_order * k) // 8)
    for k in range(8)
]


#
# Helpers
#
def FQP_point_to_FQ2_point(pt: Tuple[FQP, FQP, FQP]) -> Tuple[FQ2, FQ2, FQ2]:
    """
    Transform FQP to FQ2 for type hinting.
    """
    return (
        FQ2(pt[0].coeffs),
        FQ2(pt[1].coeffs),
        FQ2(pt[2].coeffs),
    )


def modular_squareroot(value: FQ2) -> FQP:
    """
    ``modular_squareroot(x)`` returns the value ``y`` such that ``y**2 % q == x``,
    and None if this is not possible. In cases where there are two solutions,
    the value with higher imaginary component is favored;
    if both solutions have equal imaginary component the value with higher real
    component is favored.
    """
    candidate_squareroot = value ** ((FQ2_order + 8) // 16)
    check = candidate_squareroot ** 2 / value
    if check in eighth_roots_of_unity[::2]:
        x1 = candidate_squareroot / eighth_roots_of_unity[eighth_roots_of_unity.index(check) // 2]
        x2 = FQ2([-x1.coeffs[0], -x1.coeffs[1]])  # x2 = -x1
        return x1 if (x1.coeffs[1], x1.coeffs[0]) > (x2.coeffs[1], x2.coeffs[0]) else x2
    return None


def _get_x_coordinate(message_hash: Hash32, domain: int) -> FQ2:
    domain_in_bytes = domain.to_bytes(8, 'big')

    # Initial candidate x coordinate
    x_re = big_endian_to_int(hash_eth2(message_hash + domain_in_bytes + b'\x01'))
    x_im = big_endian_to_int(hash_eth2(message_hash + domain_in_bytes + b'\x02'))
    x_coordinate = FQ2([x_re, x_im])  # x_re + x_im * i

    return x_coordinate


def hash_to_G2(message_hash: Hash32, domain: int) -> Tuple[FQ2, FQ2, FQ2]:
    x_coordinate = _get_x_coordinate(message_hash, domain)

    # Test candidate y coordinates until a one is found
    while 1:
        y_coordinate_squared = x_coordinate ** 3 + FQ2([4, 4])  # The curve is y^2 = x^3 + 4(i + 1)
        y_coordinate = modular_squareroot(y_coordinate_squared)
        if y_coordinate is not None:  # Check if quadratic residue found
            break
        x_coordinate += FQ2([1, 0])  # Add 1 and try again

    return multiply(
        (x_coordinate, y_coordinate, FQ2([1, 0])),
        G2_cofactor
    )


#
# G1
#
def compress_G1(pt: Tuple[FQ, FQ, FQ]) -> int:
    x, y = normalize(pt)
    return x.n + 2**383 * (y.n % 2)


def decompress_G1(pt: int) -> Tuple[FQ, FQ, FQ]:
    if pt == 0:
        return (FQ(1), FQ(1), FQ(0))
    x = pt % 2**383
    y_mod_2 = pt // 2**383
    y = pow((x**3 + b.n) % q, (q + 1) // 4, q)

    if pow(y, 2, q) != (x**3 + b.n) % q:
        raise ValueError(
            "he given point is not on G1: y**2 = x**3 + b"
        )
    if y % 2 != y_mod_2:
        y = q - y
    return (FQ(x), FQ(y), FQ(1))


def G1_to_pubkey(pt: int) -> BLSPubkey:
    return BLSPubkey(pt.to_bytes(48, "big"))


def pubkey_to_G1(pubkey: BLSPubkey) -> int:
    return big_endian_to_int(pubkey)

#
# G2
#


def compress_G2(pt: Tuple[FQP, FQP, FQP]) -> Tuple[int, int]:
    if not is_on_curve(pt, b2):
        raise ValueError(
            "The given point is not on the twisted curve over FQ**2"
        )
    x, y = normalize(pt)
    return (
        int(x.coeffs[0] + 2**383 * (y.coeffs[0] % 2)),
        int(x.coeffs[1])
    )


def decompress_G2(p: Tuple[int, int]) -> Tuple[FQP, FQP, FQP]:
    x1 = p[0] % 2**383
    y1_mod_2 = p[0] // 2**383
    x2 = p[1]
    x = FQ2([x1, x2])
    if x == FQ2([0, 0]):
        return FQ2([1, 0]), FQ2([1, 0]), FQ2([0, 0])
    y = modular_squareroot(x**3 + b2)
    if y is None:
        raise ValueError("Failed to find a modular squareroot")
    if y.coeffs[0] % 2 != y1_mod_2:
        y = FQ2((y * -1).coeffs)
    if not is_on_curve((x, y, FQ2([1, 0])), b2):
        raise ValueError(
            "The given point is not on the twisted curve over FQ**2"
        )
    return x, y, FQ2([1, 0])


def G2_to_signature(pt: Tuple[int, int]) -> BLSSignature:
    return BLSSignature(
        pt[0].to_bytes(48, "big") +
        pt[1].to_bytes(48, "big")
    )


def signature_to_G2(signature: BLSSignature) -> Tuple[int, int]:
    return (big_endian_to_int(signature[:48]), big_endian_to_int(signature[48:]))
