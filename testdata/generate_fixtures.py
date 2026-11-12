#!/usr/bin/env python
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Regenerate the molecular integral fixtures as FCIDUMP + keyed `meta.txt`.

Run with a pyscf env, from anywhere:

    <pyscf-python> testdata/generate_fixtures.py [name ...]

With no arguments every system is regenerated; naming one or more regenerates only those.

# Why one format

These fixtures used to ship as a `meta.txt` + `h1.dat` + `h2.dat` triple — a *dense positional*
dump of the full d^2 and d^4 MO tensors, inherited from a sibling reference package's test data
and vendored verbatim. That format has no threshold concept, so it carried every entry down to
~1e-22, while an FCIDUMP is written with a tolerance (PySCF's default, and ours, is 1e-15). The
two therefore described the same molecule with materially different content: at N2/cc-pVDZ, 66%
of `h2.dat` was below 1e-15 and 17.5% of the assembled operator's dispatch triples were terms a
packed dump would never have carried. Consolidating on FCIDUMP removes that asymmetry entirely
rather than papering over it with a reader-side knob, and costs 6-16x fewer bytes for the same
information.

# Conventions

Every dump is written by `pyscf.tools.fcidump.from_scf(..., tol=1e-15)`: 1-based indices, spatial
orbitals, chemists' `(ij|kl)`, 8-fold packed, core energy on the `0 0 0 0` sentinel line. The
reader is `qntt::integrals::load_fcidump`.

`meta.txt` is the **keyed** `key value...` dialect (`#` comments), the same one the point-group
fixtures use — NOT the positional `d nelec ms2 e_nuc e_hf e_ref` line the `.dat` triples had. A
positional file cannot grow a field without silently breaking every reader, and these metas carry
per-fixture keys (which correlation method the reference energy came from, the symmetry census)
that the positional form has no room for. `qntt::integrals::read_meta` reads only the keyed form.

The reference energy is named by its own key — `e_fci` when the determinant count is tractable,
`e_ccsd_t` otherwise — never a generic `e_ref` slot whose meaning varies per fixture. That was a
real defect in the old fixtures: two of them recorded `e_ref == e_hf` because FCI had been skipped
and the writer fell back to the SCF energy, which makes a "% of correlation energy recovered"
readout divide by zero.

# Symmetry

Every system is generated in its **abelian point group**, forced explicitly rather than left to
`symmetry=True`: for a linear molecule PySCF assigns the full Coov/Dooh, which
`pyscf.tools.fcidump._convert_orbsym` then reduces with an `orbsym % 10` fallback and a warning.
Forcing removes that dependency, and the script *verifies* it costs nothing by running the
automatic-symmetry SCF too and asserting identical ORBSYM and energy.

`n2_ccpvdz` is the one deliberate exception: it is generated with `symmetry=False` (C1). It exists
to be the symmetry-free counterpart of `testdata/n2_d2h/` at the same geometry and basis, which is
what makes the two cross-checkable (`n2_d2h/meta.txt` records the agreement of their RHF energies).
Turning symmetry on here would make it a byte-for-byte duplicate of that fixture and destroy the
only C1-vs-D2h pair in the tree.

A symmetry-adapted dump carries genuine forbidden-entry noise (entries whose orbital irreps do not
fuse to the identity, ~1e-14). That is expected and is exactly what `screen_point_group_forbidden`
exists for; the census at the bottom of each meta records it, and the script asserts every such
entry is below the reader's 1e-8 rejection floor. Callers that build a C1 space
(`PointGroupPolicy::Ignore`, the default) are unaffected either way.

NEVER hand-edit a generated file — regenerate. Every self-check runs BEFORE a byte is written.
"""

import hashlib
import json
import math
import os
import sys

import numpy as np
import pyscf
from pyscf import cc, fci, gto, mcscf, scf, symm
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))

# numpy-on-Accelerate (macOS ARM, numpy 2.0.x) raises SPURIOUS divide-by-zero/invalid FPE flags
# from dgemm on fully finite inputs; measured 2026-08-12: `mo.T @ hcore @ mo` warns while agreeing
# with einsum to 0.0. Silence the false alarm — every self-check below asserts finiteness/values.
np.seterr(divide="ignore", invalid="ignore")

WRITE_TOL = 1e-15  # PySCF's own default; the level at which a packed dump carries information
FCI_DET_CAP = 2_000_000  # above this, the reference energy is CCSD(T) instead
FORBIDDEN_FLOOR = 1e-8  # qntt's SYMMETRY_FORBIDDEN_TOL — above this a forbidden entry is an error

# Water / formaldehyde / ethylene: the coordinates of Table S1 of the CASSCF excited-states
# supplementary (Grazioli, Hu, Nottoli, Lipparini, Cances, "Critical point search and linear
# response theory ... Part II: CASSCF"), transcribed VERBATIM.
#
# Two traps, both of which the superseded gen_paper.py fell into:
#   * The table MIXES UNITS -- water and ethylene are in Bohr, formaldehyde in Angstrom. The
#     coordinates below are carried over unconverted, each with its own `unit`, so there is no
#     conversion to get wrong.  Formaldehyde and ethylene are transcribed VERBATIM and are exactly
#     symmetric (their table entries agree to 12 digits).
#
# WATER IS THE ONE DEVIATION, and it is deliberate.  Table S1(a)'s 8-decimal Bohr coordinates give
# r(OH) = 1.000000209223 and 1.000000208063 A -- they differ by 1.16e-9 A, so the transcribed
# molecule is not exactly C2v.  That breaks the symmetry-adapted SCF at ~5.4e-9 in the forbidden
# one-body integrals: below qntt's 1e-8 rejection floor, but only by a factor of 2, which is far
# too little margin to ship.  The intended geometry is unambiguous -- r = 1.0000 A and
# HOH = 104.5000 deg, the residual 2.1e-7 A being their Bohr/Angstrom conversion factor -- so water
# is built from those exact internal coordinates in a symmetric frame instead.  Same molecule to
# 1e-7 A; exactly C2v.
#   * gen_paper.py used transcribed idealized internal coordinates instead, and matched neither the
#     paper nor its own comment: it shipped r(OH)=0.9686 / 104.00 deg where the paper has
#     1.0000 / 104.50, r(CH)=1.1050 / HCO=121.57 where the paper has 1.1160 / 121.75, and
#     r(CC)=1.3390 r(CH)=1.0831 HCH=116.80 where the paper has 1.3338 / 1.0804 / 117.23. Any
#     comparison of those fixtures against the paper's tables was invalid.
#
# LiH is a small smoke fixture near equilibrium; N2 is at 2.118 bohr, matching testdata/n2_d2h/.
# Neither comes from that paper.
SYSTEMS = {
    "lih_sto3g": dict(
        molecule="LiH", dump="LIH.FCIDUMP", basis="sto-3g", unit="Angstrom", group="C2v",
        geometry="Li 0 0 0; H 0 0 1.595",
    ),
    "h2o_sto3g": dict(
        source="CASSCF-part-II supplementary Table S1(a), symmetrized to exact C2v "
                 "(r=1.0 A, HOH=104.5 deg)",
        molecule="H2O", dump="H2O.FCIDUMP", basis="sto-3g", unit="Angstrom", group="C2v",
        geometry="O 0.0 0.0 0.0; H 0.7906895737438433 0.0 0.6122172800344493; H -0.7906895737438433 0.0 0.6122172800344493",
    ),
    "h2o_631g": dict(
        source="CASSCF-part-II supplementary Table S1(a), symmetrized to exact C2v "
                 "(r=1.0 A, HOH=104.5 deg)",
        molecule="H2O", dump="H2O.FCIDUMP", basis="6-31g", unit="Angstrom", group="C2v",
        geometry="O 0.0 0.0 0.0; H 0.7906895737438433 0.0 0.6122172800344493; H -0.7906895737438433 0.0 0.6122172800344493",
    ),
    # Symmetric 2.5x bond stretch of h2o_sto3g (same C2v frame and 104.5 deg angle, r ~ 2.5 A).
    # Purpose WAS a SMALL-SCALE RPA instability: stretch into the static-correlation regime so the
    # symmetric fixed-rank minimum forks into a real-saddle/complex-minimum pair at low rank, in
    # d=7 where the dense symplectic driver runs in milliseconds. MEASURED NEGATIVE (2026-08-13/14,
    # h2o_fork_probe): 1.8x is H^ii-stable but SOFT (dense ω1 down to 5.5e-3 at M=5), and at
    # 2.5x the H^ii mode softens only to ~1.8e-3 and NEVER crosses zero — water's symmetry breaking
    # is SPATIAL and the minimal basis gives the broken solution no room to drop below the
    # symmetric one. Kept as the stable-but-soft small arm (RHF stays converged, FCI ndet=441).
    "h2o_sto3g_stretch": dict(
        source="h2o_sto3g geometry, symmetric 2.5x bond stretch (r~2.5 A, HOH=104.5 deg)",
        molecule="H2O", dump="H2O.FCIDUMP", basis="sto-3g", unit="Angstrom", group="C2v",
        geometry="O 0.0 0.0 0.0; H 1.9767239343596083 0.0 1.5305432000861233; "
                 "H -1.9767239343596083 0.0 1.5305432000861233",
    ),
    # Symmetric 2.0x bond stretch of h2o_631g. The 6-31G companion of h2o_sto3g_stretch: the split
    # valence gives the broken-symmetry solution room to drop below the symmetric one (STO-3G's
    # minimal basis does not — measured: its H^ii mode softens to ~1.8e-3 under a 2.5x stretch but
    # never crosses zero), which is why H7 found the real H2O RPA instability only at 6-31G. FCI
    # reference (ndet=1.66e6 < 2e6 cap) so no CCSD divergence at the stretch.
    "h2o_631g_stretch": dict(
        source="h2o_631g geometry, symmetric 2.0x bond stretch (r~2.0 A, HOH=104.5 deg)",
        molecule="H2O", dump="H2O.FCIDUMP", basis="6-31g", unit="Angstrom", group="C2v",
        geometry="O 0.0 0.0 0.0; H 1.5813791474876866 0.0 1.2244345600688986; "
                 "H -1.5813791474876866 0.0 1.2244345600688986",
    ),
    "hcho_sto3g": dict(
        source="CASSCF-part-II supplementary Table S1(b) (Angstrom), verbatim",
        molecule="HCHO", dump="HCHO.FCIDUMP", basis="sto-3g", unit="Angstrom", group="C2v",
        geometry="C 0.000000 0.000000 0.000000; O 0.000000 1.208000 0.000000; "
                 "H -0.948993 -0.587255 0.000000; H 0.948993 -0.587255 0.000000",
    ),
    "hcho_631g": dict(
        source="CASSCF-part-II supplementary Table S1(b) (Angstrom), verbatim",
        molecule="HCHO", dump="HCHO.FCIDUMP", basis="6-31g", unit="Angstrom", group="C2v",
        geometry="C 0.000000 0.000000 0.000000; O 0.000000 1.208000 0.000000; "
                 "H -0.948993 -0.587255 0.000000; H 0.948993 -0.587255 0.000000",
    ),
    "c2h4_631g": dict(
        source="CASSCF-part-II supplementary Table S1(c) (Bohr), verbatim",
        molecule="C2H4", dump="C2H4.FCIDUMP", basis="6-31g", unit="Bohr", group="D2h",
        geometry="H 2.32345976 1.74287672 0.00000000; H -2.32345976 1.74287672 0.00000000; "
                 "H 2.32345976 -1.74287672 0.00000000; H -2.32345976 -1.74287672 0.00000000; "
                 "C 1.26026583 0.00000000 0.00000000; C -1.26026583 0.00000000 0.00000000",
    ),
    # Linear H4 chain, 2.0 A spacing, STO-3G, C1. Built as the "guaranteed" small RPA instability:
    # past the Coulson-Fischer point the spin-restricted RHF of a stretched H chain is internally
    # unstable (a SPIN / triplet breaking a minimal basis DOES capture, unlike water's spatial one).
    # MEASURED NEGATIVE (2026-08-13, h2o_fork_probe): the fixed-rank TT minimum is H^ii-STABLE
    # at every rank 1-4 even though pyscf reports the RHF->UHF instability — the correlated
    # variational minimum heals the mean-field instability at this scale. Kept as the record of
    # that negative and as a tiny (d=4, 36-det) dense-driver smoke target. C1 avoids the
    # linear-molecule Dooh symmetry assert.
    "h4_chain_stretch": dict(
        source="linear H4 chain, 2.0 A uniform spacing (Coulson-Fischer-unstable)",
        molecule="H4", dump="H4.FCIDUMP", basis="sto-3g", unit="Angstrom", group=None,
        geometry="H 0 0 -3.0; H 0 0 -1.0; H 0 0 1.0; H 0 0 3.0",
    ),
    # C1 ON PURPOSE — the symmetry-free arm of the pair with testdata/n2_d2h/. See the docstring.
    "n2_ccpvdz": dict(
        molecule="N2", dump="N2.FCIDUMP", basis="cc-pvdz", unit="Bohr", group=None,
        geometry="N 0 0 0; N 0 0 2.118",
    ),
    # The missing STO-3G rung of the CASSCF-paper set (d = 14), HF MOs like its siblings.
    "c2h4_sto3g": dict(
        source="CASSCF-part-II supplementary Table S1(c) (Bohr), verbatim",
        molecule="C2H4", dump="C2H4.FCIDUMP", basis="sto-3g", unit="Bohr", group="D2h",
        geometry="H 2.32345976 1.74287672 0.00000000; H -2.32345976 1.74287672 0.00000000; "
                 "H 2.32345976 -1.74287672 0.00000000; H -2.32345976 -1.74287672 0.00000000; "
                 "C 1.26026583 0.00000000 0.00000000; C -1.26026583 0.00000000 0.00000000",
    ),
}

# ==================================================================================================
# CAS-MO variants (casscf-response campaign Phase 4, 2026-08-12): the SAME molecules dumped in
# CASSCF-CONVERGED orbitals via `fcidump.from_mo(..., mc.mo_coeff, molpro_orbsym=True)`, with
# `cas = (nelecas, ncas)` from the paper. Everything about a `_cas` fixture is generated in ONE
# process — the FCIDUMP, the meta, and the dipole `.prop` files — because MO column phases are not
# reproducible across runs (the F13/F14 trap): a `phase_fingerprint` key (sha256 of the rounded MO
# matrix, 16 hex chars) is written into `meta.txt` AND every `.prop` header so the Rust loader can
# refuse a mismatched pair. The converged CASSCF energy is asserted against
# `casscf_paper_references.json`'s `casscf_gs` (same geometry source, generated independently) at
# 5e-8 — a cross-file, cross-pyscf-version coherence gate that runs before a byte is written.
#
# pyscf's CASSCF canonicalization keeps the mo_coeff columns in [core | active | virtual] blocks
# (asserted below via mc.ncore/mc.ncas), which is exactly the cores-first/actives-next/virtuals-
# last ordering `qntt::solvers::joint::cas::CasPartition` requires.
#
# ★ `_cas` fixtures are C1 ON PURPOSE (`group=None`), matching `generate_casscf_references.py`'s
# own environment exactly. Measured (2026-08-12): the SYMMETRY-ADAPTED CASSCF at hcho/6-31G
# converges to a symmetry-restricted stationary point 1.78e-3 Ha ABOVE the C1 solution the JSON
# (and the paper: gs −113.849431) records — the coherence gate below caught it. A C1 CASSCF can
# rotate freely and lands on the paper's solution; the campaign's response runs are untyped
# (`PointGroup::TRIVIAL`) regardless, so the `_cas` dumps need no ORBSYM.
# `cas_sort_mo` (1-based MO indices for `mcscf.sort_mo`) steers the CASSCF into the paper's
# basin where the default HOMO-window guess converges elsewhere. Measured (2026-08-12, pyscf
# 2.13.1): hcho's default window lands 1.78e-3 Ha above the JSON solution; a scan over
# 2-occupied + 1-virtual windows finds exactly {5,7} + {9} at the JSON value (dev −5.6e-11) —
# the n/π occupied pair plus π*, not the two HOMOs the default picks.
for _name, _base, _cas, _sort in [
    ("h2o_sto3g_cas", "h2o_sto3g", (8, 6), None),
    ("h2o_631g_cas", "h2o_631g", (8, 6), None),
    ("hcho_631g_cas", "hcho_631g", (4, 3), [5, 7, 9]),
    ("c2h4_631g_cas", "c2h4_631g", (2, 2), None),
]:
    SYSTEMS[_name] = dict(SYSTEMS[_base], cas=_cas, props=True, group=None)
    if _sort is not None:
        SYSTEMS[_name]["cas_sort_mo"] = _sort


def molpro_to_xor(group):
    """Molpro id − 1 -> XOR index, inverted from PySCF's own `ORBSYM_MAP` (pyscf id -> molpro id).

    The same single source of truth `testdata/pointgroups/generate.py` and
    `testdata/n2_d2h/generate.py` use; never re-derived by hand.
    """
    pyscf_to_molpro = fcidump.ORBSYM_MAP[group]
    out = [None] * len(pyscf_to_molpro)
    for pyscf_id, molpro_id in enumerate(pyscf_to_molpro):
        out[molpro_id - 1] = pyscf_id
    assert all(x is not None for x in out), out
    return out


def assert_symmetry_adapted(xor, h1e, eri, tol=1e-10):
    """Every integral whose orbital irreps do not fuse to the identity must be numerical noise.

    Returns the worst one-body and two-body violation, for the record in `meta.txt`.
    """
    x = np.asarray(xor, dtype=np.int64)
    bad1 = np.abs(h1e[(x[:, None] ^ x[None, :]) != 0])
    m1 = float(bad1.max()) if bad1.size else 0.0
    assert m1 < tol, f"one-body symmetry violation {m1:e}"
    f4 = x[:, None, None, None] ^ x[None, :, None, None] ^ x[None, None, :, None] ^ x[None, None, None, :]
    bad2 = np.abs(eri[f4 != 0])
    m2 = float(bad2.max()) if bad2.size else 0.0
    assert m2 < tol, f"two-body symmetry violation {m2:e}"
    return m1, m2


def census(path, group, orbsym_molpro):
    """Forbidden-entry census of the SHIPPED bytes, in both conventions.

    `*_lines` counts raw file lines (the unique 8-fold triangle PySCF writes); the unsuffixed keys
    count distinct tensor CELLS after `TwoBodySymmetry::Eightfold` completion — what the Rust
    reader actually holds, and what the screen walks.
    """
    if group is None:
        return None
    xor = molpro_to_xor(group)
    x = [xor[i - 1] for i in orbsym_molpro]

    ones, twos = [], []
    for line in open(path):
        tok = line.split()
        if len(tok) != 5:
            continue
        try:
            val = float(tok[0].replace("D", "E").replace("d", "E"))
        except ValueError:
            continue
        try:
            i, j, k, l = (int(t) for t in tok[1:])
        except ValueError:
            continue
        if i == j == k == l == 0:
            continue
        if k == 0 and l == 0:
            ones.append((val, (i, j)))
        else:
            twos.append((val, (i, j, k, l)))

    n1l = sum(1 for v, (i, j) in ones if (x[i - 1] ^ x[j - 1]) and v != 0.0)
    m1l = max((abs(v) for v, (i, j) in ones if (x[i - 1] ^ x[j - 1])), default=0.0)
    n2l = sum(1 for v, (i, j, k, l) in twos
              if (x[i - 1] ^ x[j - 1] ^ x[k - 1] ^ x[l - 1]) and v != 0.0)
    m2l = max((abs(v) for v, (i, j, k, l) in twos
               if (x[i - 1] ^ x[j - 1] ^ x[k - 1] ^ x[l - 1])), default=0.0)

    tcells, vcells = {}, {}
    for val, (i, j) in ones:
        tcells.setdefault((i - 1, j - 1), val)
        tcells.setdefault((j - 1, i - 1), val)
    for val, (i, j, k, l) in twos:
        for im in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                   (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
            vcells.setdefault(tuple(t - 1 for t in im), val)

    n1c, m1c = 0, 0.0
    n2c, m2c = 0, 0.0
    for (i, j), v in tcells.items():
        if (x[i] ^ x[j]) and v != 0.0:
            n1c += 1
            m1c = max(m1c, abs(v))
    for (i, j, k, l), v in vcells.items():
        if (x[i] ^ x[j] ^ x[k] ^ x[l]) and v != 0.0:
            n2c += 1
            m2c = max(m2c, abs(v))
    return dict(lines=(n1l, m1l, n2l, m2l), cells=(n1c, m1c, n2c, m2c),
                n_cells=(len(tcells), len(vcells)))


def ndet(norb, nelec, spin=0):
    na = (nelec + spin) // 2
    nb = (nelec - spin) // 2
    return math.comb(norb, na) * math.comb(norb, nb)


def phase_fingerprint(mo):
    """16 hex chars identifying THIS MO matrix, column phases included.

    MO phases (and near-degenerate rotations) are not reproducible across runs — the F13/F14
    trap — so an FCIDUMP and a property file are a matched pair only if they were transformed by
    the same matrix in the same process. Rounding to 10 decimals keeps the fingerprint stable
    against pure-roundoff drift while any phase flip (a sign is a factor of −1, far above 1e-10)
    changes it.
    """
    h = hashlib.sha256()
    mo = np.ascontiguousarray(np.round(np.asarray(mo, dtype=np.float64), 10))
    h.update(str(mo.shape).encode())
    h.update(mo.tobytes())
    return h.hexdigest()[:16]


def paper_casscf_gs(molecule, basis):
    """`casscf_gs` from casscf_paper_references.json — the cross-file coherence gate."""
    with open(os.path.join(HERE, "casscf_paper_references.json")) as f:
        refs = json.load(f)
    key = {"H2O": "h2o", "HCHO": "hcho", "C2H4": "c2h4"}[molecule]
    return float(refs[key]["bases"][basis]["casscf_gs"])


def write_property(outdir, comp, m, fingerprint, tol=1e-15):
    """One dipole component in the `parse_property` keyed format (1-based, FULL matrix —
    the reader completes nothing)."""
    d = m.shape[0]
    lines = [
        f"# testdata dipole component, generated by ../generate_fixtures.py in the SAME process",
        f"# as the FCIDUMP it pairs with. DO NOT EDIT; regenerate.",
        f"label dipole_{comp}",
        "unit bohr",
        "origin 0.0 0.0 0.0",
        f"phase_fingerprint {fingerprint}",
    ]
    for p in range(d):
        for q in range(d):
            if abs(m[p, q]) >= tol:
                # `float(...)` is load-bearing: numpy 2.x scalar repr is `np.float64(...)`,
                # which the record scanner (rightly) skips as a non-record — a silently EMPTY
                # property matrix (caught by tests/cas_fixtures.rs's non-vacuity arm).
                lines.append(f"{float(m[p, q])!r}  {p + 1}  {q + 1}")
    path = os.path.join(outdir, f"dipole_{comp}.prop")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def generate(name, spec):
    group, geom, basis, unit = spec["group"], spec["geometry"], spec["basis"], spec["unit"]
    outdir = os.path.join(HERE, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n===== {name}  ({spec['molecule']} / {basis} / {group or 'C1'}) =====", flush=True)

    mol = gto.M(atom=geom, basis=basis, unit=unit,
                symmetry=(group if group else False), verbose=0)
    if group:
        assert mol.groupname == group, (mol.groupname, group)
    mf = scf.RHF(mol).run()
    assert mf.converged, f"{name}: RHF did not converge"
    norb, nelec = int(mol.nao), int(mol.nelectron)
    e_nuc, e_hf = float(mf.energy_nuc()), float(mf.e_tot)
    print(f"  norb={norb} nelec={nelec}  E_HF={e_hf:.10f}", flush=True)

    # --- CAS-MO path: converge CASSCF and dump ITS orbitals (see the SYSTEMS comment) ---
    mc = None
    if "cas" in spec:
        nelecas, ncas = spec["cas"]
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        if "cas_sort_mo" in spec:
            mc.kernel(mcscf.sort_mo(mc, mf.mo_coeff, spec["cas_sort_mo"]))
        else:
            mc.kernel()
        assert mc.converged, f"{name}: CASSCF({nelecas},{ncas}) did not converge"
        ref_gs = paper_casscf_gs(spec["molecule"], basis)
        dev = abs(float(mc.e_tot) - ref_gs)
        assert dev < 5e-8, (
            f"{name}: CASSCF e_tot {mc.e_tot!r} disagrees with casscf_paper_references.json "
            f"casscf_gs {ref_gs!r} by {dev:e} — same geometry source, so this is a real "
            f"incoherence, not tolerance"
        )
        # pyscf keeps mo_coeff in [core | active | virtual] blocks — the ordering CasPartition
        # requires. ncore/ncas are pyscf's own; record, and sanity-check the electron count.
        assert int(mc.ncore) * 2 + nelecas == nelec, (name, mc.ncore, nelecas, nelec)
        print(
            f"  CASSCF({nelecas},{ncas}): E={float(mc.e_tot)!r}  (JSON casscf_gs dev {dev:.1e})"
            f"  ncore={int(mc.ncore)} ncas={int(mc.ncas)}",
            flush=True,
        )
    mo_dump = np.asarray(mc.mo_coeff if mc is not None else mf.mo_coeff)
    fingerprint = phase_fingerprint(mo_dump)

    orbsym_pyscf = orbsym_molpro = irrep_names = None
    noise1 = noise2 = None
    if group:
        if mc is not None:
            orbsym_pyscf = [
                int(s)
                for s in symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_dump)
            ]
        else:
            orbsym_pyscf = [int(s) for s in mf.mo_coeff.orbsym]
        orbsym_molpro = [int(fcidump.ORBSYM_MAP[group][i]) for i in orbsym_pyscf]
        order = len(fcidump.ORBSYM_MAP[group])
        assert set(orbsym_molpro) <= set(range(1, order + 1)), orbsym_molpro
        # Non-vacuity: an all-1s ORBSYM would be a C1 dump wearing a costume.
        assert len(set(orbsym_molpro)) > 1, f"{name}: ORBSYM is trivial — {orbsym_molpro}"
        irrep_names = [symm.irrep_id2name(group, i) for i in orbsym_pyscf]

        # Forcing the abelian group must agree with PySCF's automatic assignment (which may pick
        # the full Coov/Dooh and reduce by `orbsym % 10` inside `_convert_orbsym`). Assert it
        # rather than claim it in prose.
        mol_auto = gto.M(atom=geom, basis=basis, unit=unit, symmetry=True, verbose=0)
        mf_auto = scf.RHF(mol_auto).run()
        assert mf_auto.converged
        auto = [int(fcidump.ORBSYM_MAP[group][i])
                for i in (np.asarray(mf_auto.mo_coeff.orbsym) % 10)]
        # Compare the PARTITION, not the labels. Which of the two mirror planes is called b1 and
        # which b2 depends on the frame the molecule is presented in, and relabelling them is an
        # automorphism of C2v -- it permutes irrep NAMES and changes no number (the same reason
        # `PointGroupKind::default_for_order`'s guess is documented as name-only). Demanding
        # identical labels would fail on any molecule not already in PySCF's canonical frame.
        same = lambda v: {(i, j) for i in range(len(v)) for j in range(len(v)) if v[i] == v[j]}
        # The auto-vs-forced comparison validates the GROUP FORCING at the RHF level, so it always
        # compares against the RHF orbsym — on the CAS-MO path `orbsym_molpro` is CASSCF-based and
        # its index partition legitimately differs (CASSCF reorders columns within blocks).
        hf_orbsym_molpro = [
            int(fcidump.ORBSYM_MAP[group][int(s)]) for s in mf.mo_coeff.orbsym
        ]
        assert same(auto) == same(hf_orbsym_molpro), (name, auto, hf_orbsym_molpro)
        assert abs(mf_auto.e_tot - e_hf) < 1e-10, (mf_auto.e_tot, e_hf)

        h1e = mo_dump.T @ np.asarray(mf.get_hcore()) @ mo_dump
        eri = np.asarray(mol.ao2mo(mo_dump, aosym=1)).reshape(norb, norb, norb, norb)
        xor = molpro_to_xor(group)
        noise1, noise2 = assert_symmetry_adapted([xor[i - 1] for i in orbsym_molpro], h1e, eri)
        print(f"  MO symmetry noise: 1-body {noise1:e}  2-body {noise2:e}", flush=True)

    # --- reference energy: exact where the determinant count allows, CCSD(T) otherwise ---
    nd = ndet(norb, nelec)
    if nd <= FCI_DET_CAP:
        ref_key, ref_val = "e_fci", float(np.atleast_1d(fci.FCI(mf).kernel()[0])[0])
        extra = {}
        print(f"  ndet={nd:.3e} -> FCI  E={ref_val:.10f}", flush=True)
    else:
        mycc = cc.CCSD(mf).run()
        assert mycc.converged, f"{name}: CCSD did not converge"
        e_ccsd = float(mycc.e_tot)
        ref_key, ref_val = "e_ccsd_t", float(mycc.e_tot + mycc.ccsd_t())
        extra = {"e_ccsd": e_ccsd}
        print(f"  ndet={nd:.3e} -> CCSD(T)  E_CCSD={e_ccsd:.10f} E_CCSD(T)={ref_val:.10f}",
              flush=True)

    # --- write (nothing above here has touched the disk) ---
    path = os.path.join(outdir, spec["dump"])
    if mc is not None:
        # CAS-MO dump: the CASSCF orbitals, with THEIR irrep labels. `from_mo` computes the same
        # hcore/eri transforms and the nuclear core energy `from_scf` would.
        # ★ A caller-supplied `orbsym` is written VERBATIM (`molpro_orbsym` only applies to the
        # `orbsym is None` fallback path — read the pyscf source, the round-trip assert below
        # caught exactly this), so pass the already-converted MOLPRO ids.
        fcidump.from_mo(mol, path, mo_dump, orbsym=orbsym_molpro, tol=WRITE_TOL)
    elif group:
        fcidump.from_scf(mf, path, tol=WRITE_TOL, molpro_orbsym=True)
    else:
        fcidump.from_scf(mf, path, tol=WRITE_TOL)

    # --- post-write round-trip against the shipped bytes ---
    back = fcidump.read(path, verbose=False)
    assert int(back["NORB"]) == norb and int(back["NELEC"]) == nelec and int(back["MS2"]) == 0
    assert abs(float(back["ECORE"]) - e_nuc) < 1e-12
    if group:
        assert list(int(s) for s in back["ORBSYM"]) == orbsym_molpro
    rep = census(path, group, orbsym_molpro)
    if rep is not None:
        n1c, m1c, n2c, m2c = rep["cells"][0], rep["cells"][1], rep["cells"][2], rep["cells"][3]
        assert max(m1c, m2c) < FORBIDDEN_FLOOR, (name, rep["cells"])
        print(f"  forbidden cells: 1-body {n1c} (max {m1c:e}), 2-body {n2c} (max {m2c:e})",
              flush=True)

    # --- meta ---
    lines = [
        f"# testdata/{name}/meta.txt -- generated by ../generate_fixtures.py, DO NOT EDIT.",
        "# Keyed `key value...` (NOT the positional `d nelec ms2 e_nuc e_hf e_ref` the superseded",
        "# .dat fixtures used): this file is self-describing and a positional one would break on an",
        "# inserted field.  All floats are Python `repr` (shortest round-tripping).",
        f"pyscf_version   {pyscf.__version__}",
        f"molecule        {spec['molecule']}",
        f"geometry        {geom}",
        f"geometry_source {spec.get('source', 'near-equilibrium, not from a published table')}",
        f"unit            {unit}",
        f"basis           {basis}",
        f"point_group     {group or 'C1'}",
    ]
    if group:
        lines += [
            f"top_group       {mol.topgroup}",
            f"orbsym_molpro   {' '.join(str(s) for s in orbsym_molpro)}",
            f"orbsym_pyscf    {' '.join(str(s) for s in orbsym_pyscf)}",
            f"irrep_names     {' '.join(irrep_names)}",
        ]
    elif mc is not None:
        lines += [
            "# C1 ON PURPOSE: the `_cas` fixtures match generate_casscf_references.py's own",
            "# (symmetry-free) environment — a symmetry-adapted CASSCF can converge to a",
            "# symmetry-restricted stationary point above the paper's solution (measured 1.78e-3",
            "# at hcho/6-31G, caught by the casscf_gs coherence gate).",
        ]
    elif name == "n2_ccpvdz":
        lines += [
            "# symmetry=False ON PURPOSE: this is the C1 arm of the pair with testdata/n2_d2h/",
            "# (same geometry and basis), which is what makes the two cross-checkable.",
        ]
    else:
        lines += [
            "# symmetry=False ON PURPOSE (see this system's entry in generate_fixtures.py).",
        ]
    lines += [
        f"norb            {norb}",
        f"nelec           {nelec}",
        "ms2             0",
        "isym            1",
        f"e_nuc           {e_nuc!r}",
        f"e_hf            {e_hf!r}",
    ]
    lines += [
        "# Which orbitals the dump is expressed in, and the fingerprint that ties every",
        "# same-process property file to THIS MO matrix (phases are not reproducible across runs).",
        f"mo_source       {'casscf' if mc is not None else 'rhf'}",
        f"phase_fingerprint {fingerprint}",
    ]
    if mc is not None:
        lines += [
            "# The CAS(n,m) partition of the CASSCF these orbitals come from; columns are",
            "# [core | active | virtual] contiguous, cores first (pyscf canonicalization).",
            f"cas_nelec       {spec['cas'][0]}",
            f"cas_norb        {spec['cas'][1]}",
            f"n_core          {int(mc.ncore)}",
            f"n_active        {int(mc.ncas)}",
            f"n_virt          {norb - int(mc.ncore) - int(mc.ncas)}",
            f"casscf_gs       {float(mc.e_tot)!r}",
        ]
    for k, v in extra.items():
        lines.append(f"{k}{' ' * (16 - len(k))}{v!r}")
    lines += [
        f"# Reference (correlated) energy.  `{ref_key}` names the method: FCI where the determinant",
        f"# count allows (<= {FCI_DET_CAP:g}), CCSD(T) otherwise.  Never a generic `e_ref` slot.",
        f"{ref_key}{' ' * (16 - len(ref_key))}{ref_val!r}",
        f"ndet            {nd}",
        f"write_tol       {WRITE_TOL!r}",
        f"dump_file       {spec['dump']}",
        f"dump_bytes      {os.path.getsize(path)}",
        f"dump_lines      {sum(1 for _ in open(path))}",
    ]
    if rep is not None:
        n1l, m1l, n2l, m2l = rep["lines"]
        n1c, m1c, n2c, m2c = rep["cells"]
        lines += [
            "# Forbidden-entry census of the SHIPPED bytes, two conventions: `*_lines` are raw file",
            "# lines (the unique 8-fold triangle PySCF writes); the unsuffixed keys are distinct",
            "# t/v tensor CELLS after TwoBodySymmetry::Eightfold completion.  All magnitudes are",
            f"# decades below the reader's {FORBIDDEN_FLOOR:g} rejection floor: this is the silently-",
            "# screened symmetry-noise regime, never an Err.",
            f"forbidden_1body_lines {n1l}",
            f"forbidden_2body_lines {n2l}",
            f"forbidden_max_lines   {max(m1l, m2l)!r}",
            f"forbidden_1body {n1c}",
            f"forbidden_2body {n2c}",
            f"forbidden_1body_max {m1c!r}",
            f"forbidden_2body_max {m2c!r}",
            f"forbidden_max   {max(m1c, m2c)!r}",
            "# Largest symmetry-forbidden magnitude in the in-memory MO integrals, before PySCF's",
            "# write threshold dropped anything.",
            f"mo_noise_1body  {noise1!r}",
            f"mo_noise_2body  {noise2!r}",
            f"cells_1body     {rep['n_cells'][0]}",
            f"cells_2body     {rep['n_cells'][1]}",
        ]
    with open(os.path.join(outdir, "meta.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path} ({os.path.getsize(path)} bytes) + meta.txt", flush=True)

    # --- properties (same process, same MO matrix — that is the whole point) ---
    if spec.get("props"):
        r_ao = mol.intor("int1e_r")  # (3, nao, nao), AO basis, origin at (0,0,0)
        for comp, ao in zip("xyz", r_ao):
            m = mo_dump.T @ np.asarray(ao) @ mo_dump
            asym = float(np.abs(m - m.T).max())
            assert asym < 1e-12, (name, comp, asym)
            ppath = write_property(outdir, comp, m, fingerprint)
            print(f"  wrote {ppath} ({os.path.getsize(ppath)} bytes)", flush=True)


def main():
    wanted = sys.argv[1:] or list(SYSTEMS)
    unknown = [w for w in wanted if w not in SYSTEMS]
    assert not unknown, f"unknown system(s): {unknown}; known: {list(SYSTEMS)}"
    print(f"pyscf {pyscf.__version__}, numpy {np.__version__}, write_tol={WRITE_TOL:g}")
    for name in wanted:
        generate(name, SYSTEMS[name])


if __name__ == "__main__":
    main()
