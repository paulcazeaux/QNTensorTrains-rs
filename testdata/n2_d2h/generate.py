#!/usr/bin/env python
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Generate the PG6 symmetry-adapted N2 / cc-pVDZ (D2h) FCIDUMP fixture — the perf-gate scale.

Run with the pyscf env, from anywhere:
    ~/miniforge3/envs/pyscf/bin/python testdata/n2_d2h/generate.py

Writes `N2.FCIDUMP` (Molpro-convention ORBSYM + ISYM, 8-fold-packed two-body block, PySCF's own
`tol=1e-15` write threshold) and `meta.txt` (PySCF's OWN reference numbers).  NEVER hand-edit
either output — regenerate.  All self-checks run BEFORE a single byte is written.

Two deliberate choices:

* **`symmetry='D2h'` is FORCED.**  `symmetry=True` assigns N2 the full `Dooh`, which
  `pyscf.tools.fcidump._convert_orbsym` then falls back to D2h with a warning and an
  `orbsym % 10` reduction.  Forcing removes that dependency; the script *verifies* it costs
  nothing by running the automatic-symmetry SCF too and asserting identical ORBSYM and energy.
* **Same geometry (2.118 bohr) as the tracked C1 fixture `testdata/n2_ccpvdz/`**, so the two are
  cross-checkable: the meta records the agreement of this dump's RHF energy with that fixture's.

FCI at (28o, 14e) is not tractable; the meta pins SCF, MP2, CCSD and CCSD(T) instead.  This dump
carries genuine symmetry noise — a few hundred entries whose orbital irreps do not fuse to the
identity, all far below the reader's 1e-8 rejection floor — which is exactly the regime the
point-group screening path exists to handle, and the reason this fixture (not H2O) is the one
that exercises it.
"""

import os

import numpy as np
import pyscf
from pyscf import cc, gto, mp, scf, symm
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = "D2h"
GEOMETRY = "N 0 0 0; N 0 0 2.118"
UNIT = "Bohr"
BASIS = "cc-pvdz"
NORB = 28
NELEC = 14
DUMP = "N2.FCIDUMP"
# testdata/n2_ccpvdz/meta.txt (symmetry-free RHF + CCSD(T) at the SAME geometry).
N2_CCPVDZ_E_HF = -108.94937787901756
N2_CCPVDZ_E_REF = -109.2803050677459


def molpro_to_xor(group):
    """Molpro id − 1 -> XOR index, inverted from PySCF's own `ORBSYM_MAP` (pyscf id -> molpro id).

    Same single source of truth as `testdata/pointgroups/generate.py`; never re-derived by hand.
    """
    pyscf_to_molpro = fcidump.ORBSYM_MAP[group]
    out = [0] * len(pyscf_to_molpro)
    for pyscf_id, molpro_id in enumerate(pyscf_to_molpro):
        out[molpro_id - 1] = pyscf_id
    return out


def assert_integrals_are_symmetry_adapted(xor, h1e, eri, tol=1e-10):
    """Certify the MOs really are adapted to `GROUP`: every integral whose orbital irreps do not
    fuse (XOR) to the identity must be numerical noise.  Checked on the in-memory MO integrals
    BEFORE writing, so a fixture can never ship claiming a symmetry it does not have."""
    x = np.asarray(xor)
    g1 = x[:, None] ^ x[None, :]
    bad1 = np.abs(h1e)[g1 != 0]
    assert bad1.size == 0 or bad1.max() < tol, f"one-body symmetry violation {bad1.max():e}"
    g2 = g1[:, :, None, None] ^ g1[None, None, :, :]
    bad2 = np.abs(eri)[g2 != 0]
    assert bad2.size == 0 or bad2.max() < tol, f"two-body symmetry violation {bad2.max():e}"
    return (0.0 if bad1.size == 0 else float(bad1.max()),
            0.0 if bad2.size == 0 else float(bad2.max()))


def read_dump_entries(path):
    """Parse a written FCIDUMP back into `(orbsym, isym, e0, one_body_lines, two_body_lines)`.

    Deliberately a *file-level* parse of the shipped bytes (not a re-use of the in-memory arrays),
    so every census below is a statement about the artifact that is actually vendored.
    """
    orbsym, isym, e0 = None, None, None
    ones, twos = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("ORBSYM="):
                orbsym = [int(x) for x in s[len("ORBSYM="):].strip(",").split(",")]
            if s.startswith("ISYM="):
                isym = int(s[len("ISYM="):].strip(","))
            if "&END" in s.upper():
                break
        for line in f:
            tok = line.split()
            if len(tok) != 5:
                continue
            x = float(tok[0])
            i, j, k, l = (int(t) for t in tok[1:])
            if i == 0 and j == 0 and k == 0 and l == 0:
                e0 = x
            elif k == 0 and l == 0:
                ones.append((x, i, j))
            else:
                twos.append((x, [i, j, k, l]))
    return orbsym, isym, e0, ones, twos


def census(path, group):
    """Forbidden-entry census of the shipped file, in BOTH conventions.

    `lines`   — raw file lines whose orbital irreps do not fuse to the identity (the unique
                8-fold triangle PySCF writes).
    `cells`   — distinct `t`/`v` tensor cells the Rust reader ends up holding, i.e. after
                `TwoBodySymmetry::Eightfold` completion.  This is what `tests/pointgroup_pg6.rs`'s
                `screen_forbidden` walks, so it is the number the Rust test asserts against; it is
                larger than the line count (up to 2x for `t`, 8x for `v`) because one written line
                fills several images.  The completion rule below mirrors `read_fcidump`
                (src/fcidump.rs): explicit entries first, then images inserted only into slots not
                already occupied.
    """
    x = molpro_to_xor(group)
    orbsym, isym, e0, ones, twos = read_dump_entries(path)
    xor = [x[i - 1] for i in orbsym]

    n1l = n2l = 0
    m1l = m2l = 0.0
    for val, i, j in ones:
        if xor[i - 1] ^ xor[j - 1]:
            n1l += 1
            m1l = max(m1l, abs(val))
    for val, idx in twos:
        if xor[idx[0] - 1] ^ xor[idx[1] - 1] ^ xor[idx[2] - 1] ^ xor[idx[3] - 1]:
            n2l += 1
            m2l = max(m2l, abs(val))

    tcells = {}
    for val, i, j in ones:
        tcells[(i - 1, j - 1)] = val
    for val, i, j in ones:
        tcells.setdefault((j - 1, i - 1), val)
    vcells = {}
    for val, idx in twos:
        vcells[tuple(t - 1 for t in idx)] = val
    for val, (i, j, k, l) in twos:
        for im in ([j, i, k, l], [i, j, l, k], [j, i, l, k], [k, l, i, j],
                   [l, k, i, j], [k, l, j, i], [l, k, j, i]):
            vcells.setdefault(tuple(t - 1 for t in im), val)

    n1c = n2c = 0
    m1c = m2c = 0.0
    for (i, j), val in tcells.items():
        if (xor[i] ^ xor[j]) and val != 0.0:
            n1c += 1
            m1c = max(m1c, abs(val))
    for (i, j, k, l), val in vcells.items():
        if (xor[i] ^ xor[j] ^ xor[k] ^ xor[l]) and val != 0.0:
            n2c += 1
            m2c = max(m2c, abs(val))

    return dict(orbsym=orbsym, isym=isym, e0=e0,
                lines=(n1l, m1l, n2l, m2l), cells=(n1c, m1c, n2c, m2c),
                n_cells=(len(tcells), len(vcells)))


def main():
    mol = gto.M(atom=GEOMETRY, basis=BASIS, unit=UNIT, symmetry=GROUP, verbose=0)
    assert mol.groupname == GROUP, mol.groupname
    assert mol.topgroup == "Dooh", mol.topgroup  # forced down from D∞h — see the module docstring
    assert mol.nelectron == NELEC and mol.spin == 0
    assert mol.nao == NORB, mol.nao

    mf = scf.RHF(mol).run()
    assert mf.converged

    orbsym_pyscf = [int(s) for s in mf.mo_coeff.orbsym]
    assert len(orbsym_pyscf) == mol.nao == NORB
    orbsym_molpro = [fcidump.ORBSYM_MAP[GROUP][i] for i in orbsym_pyscf]
    order = len(fcidump.ORBSYM_MAP[GROUP])
    assert set(orbsym_molpro) <= set(range(1, order + 1)), orbsym_molpro
    # Non-vacuity: an all-1s ORBSYM would be a C1 dump wearing a costume.
    assert len(set(orbsym_molpro)) > 1, orbsym_molpro
    irrep_names = [symm.irrep_id2name(GROUP, i) for i in orbsym_pyscf]

    # Forcing D2h must be equivalent to PySCF's automatic assignment (Dooh, reduced by ORBSYM % 10
    # inside `_convert_orbsym`) — run it and assert, rather than assert it in prose.
    mol_auto = gto.M(atom=GEOMETRY, basis=BASIS, unit=UNIT, symmetry=True, verbose=0)
    mf_auto = scf.RHF(mol_auto).run()
    assert mf_auto.converged and mol_auto.groupname == "Dooh", mol_auto.groupname
    orbsym_auto = [fcidump.ORBSYM_MAP[GROUP][i] for i in (np.asarray(mf_auto.mo_coeff.orbsym) % 10)]
    assert orbsym_auto == orbsym_molpro, (orbsym_auto, orbsym_molpro)
    assert abs(mf_auto.e_tot - mf.e_tot) < 1e-10, (mf_auto.e_tot, mf.e_tot)

    # Symmetry self-check on the MO integrals, before writing (28^4 = 614656 cells, vectorized).
    mo = np.asarray(mf.mo_coeff)
    h1e = mo.T @ np.asarray(mf.get_hcore()) @ mo
    eri = np.asarray(mol.ao2mo(mo, aosym=1)).reshape(NORB, NORB, NORB, NORB)
    xor = molpro_to_xor(GROUP)
    noise1, noise2 = assert_integrals_are_symmetry_adapted([xor[i - 1] for i in orbsym_molpro],
                                                           h1e, eri)

    e_nuc, e_hf = float(mf.energy_nuc()), float(mf.e_tot)

    # --- write ---
    path = os.path.join(HERE, DUMP)
    fcidump.from_scf(mf, path, tol=1e-15, molpro_orbsym=True)

    # --- post-write round-trip against the shipped bytes ---
    back = fcidump.read(path, verbose=False)
    assert back["NORB"] == NORB and back["NELEC"] == NELEC and back["MS2"] == 0
    assert list(back["ORBSYM"]) == orbsym_molpro, (list(back["ORBSYM"]), orbsym_molpro)
    assert int(back["ISYM"]) == 1, back["ISYM"]  # PySCF's write_head hardcodes ISYM=1
    assert abs(float(back["ECORE"]) - e_nuc) < 1e-12
    rep = census(path, GROUP)
    assert rep["orbsym"] == orbsym_molpro and rep["isym"] == 1
    # This fixture's raison d'être: it MUST carry forbidden entries, and they must all be noise.
    n1l, m1l, n2l, m2l = rep["lines"]
    n1c, m1c, n2c, m2c = rep["cells"]
    assert n1c > 0 and n2c > 0, rep["cells"]
    assert max(m1c, m2c) < 1e-8, rep["cells"]  # the reader's rejection floor

    # --- PySCF's own physics, pinned into the meta (FCI at 28o/14e is intractable) ---
    e_mp2_corr = float(mp.MP2(mf).kernel()[0])
    mycc = cc.CCSD(mf).run()
    assert mycc.converged
    e_ccsd = float(mycc.e_tot)
    e_ccsd_t = float(mycc.e_tot + mycc.ccsd_t())
    # Loose on purpose: the tracked C1 fixture ran a symmetry-free SCF/CCSD at the same geometry,
    # so the two agree only to their respective convergence thresholds (measured: 2e-13 on the
    # SCF, 1e-9 on CCSD(T) -- the perturbative triples correction is the slack one).
    assert abs(e_hf - N2_CCPVDZ_E_HF) < 1e-9, (e_hf, N2_CCPVDZ_E_HF)
    assert abs(e_ccsd_t - N2_CCPVDZ_E_REF) < 1e-7, (e_ccsd_t, N2_CCPVDZ_E_REF)

    with open(os.path.join(HERE, "meta.txt"), "w") as f:
        f.write(f"""# testdata/n2_d2h/meta.txt -- generated by generate.py, DO NOT EDIT.
# Keyed `key value...` (NOT the positional `d nelec ms2 e_nuc e_hf e_ref` of the .dat fixtures):
# PG6's fixtures are self-describing FCIDUMPs and a positional file would break on an inserted
# field.  All floats are Python `repr` (shortest round-tripping).
pyscf_version   {pyscf.__version__}
molecule        N2
geometry        {GEOMETRY}
unit            {UNIT}
basis           {BASIS}
point_group     {GROUP}
top_group       {mol.topgroup}
norb            {NORB}
nelec           {NELEC}
ms2             0
isym            1
orbsym_molpro   {" ".join(str(s) for s in orbsym_molpro)}
orbsym_pyscf    {" ".join(str(s) for s in orbsym_pyscf)}
irrep_names     {" ".join(irrep_names)}
e_nuc           {e_nuc!r}
e_hf            {e_hf!r}
e_mp2_corr      {e_mp2_corr!r}
e_mp2_tot       {e_hf + e_mp2_corr!r}
e_ccsd          {e_ccsd!r}
e_ccsd_t        {e_ccsd_t!r}
# FCI is intractable at (28o, 14e) -- no e_fci key here (contrast testdata/h2o_c2v/meta.txt).
# Cross-check against the tracked symmetry-free fixture at the SAME geometry
# (testdata/n2_ccpvdz/meta.txt: e_hf {N2_CCPVDZ_E_HF!r}, e_ref(CCSD(T)) {N2_CCPVDZ_E_REF!r}):
cross_check_n2_ccpvdz_e_hf_delta   {abs(e_hf - N2_CCPVDZ_E_HF)!r}
cross_check_n2_ccpvdz_e_ref_delta  {abs(e_ccsd_t - N2_CCPVDZ_E_REF)!r}
# Forbidden-entry census of the SHIPPED bytes, two conventions (see generate.py::census):
#   *_lines  raw file lines (the unique 8-fold triangle PySCF writes);
#   the unsuffixed keys are the distinct t/v tensor CELLS after TwoBodySymmetry::Eightfold
#   completion -- what tests/pointgroup_pg6.rs's `screen_forbidden` walks.
# All magnitudes are decades below the reader's 1e-8 rejection floor: this dump is squarely in
# the silently-screened symmetry-noise regime, never an Err.
forbidden_1body_lines {n1l}
forbidden_2body_lines {n2l}
forbidden_max_lines   {max(m1l, m2l)!r}
forbidden_1body {n1c}
forbidden_2body {n2c}
forbidden_1body_max {m1c!r}
forbidden_2body_max {m2c!r}
forbidden_max   {max(m1c, m2c)!r}
# Largest symmetry-forbidden magnitude in the in-memory MO integrals, before PySCF's 1e-15 write
# threshold dropped anything (generate.py::assert_integrals_are_symmetry_adapted).
mo_noise_1body  {noise1!r}
mo_noise_2body  {noise2!r}
# Nonzero t/v cells the Rust reader holds after Eightfold completion (context for the census).
cells_1body     {rep["n_cells"][0]}
cells_2body     {rep["n_cells"][1]}
dump_file       {DUMP}
dump_bytes      {os.path.getsize(path)}
dump_lines      {sum(1 for _ in open(path))}
""")

    print(f"{GROUP}: norb={NORB} nelec={NELEC} orbsym={orbsym_molpro}")
    print(f"names={irrep_names}")
    print(f"e_nuc={e_nuc!r} e_hf={e_hf!r}")
    print(f"e_mp2_corr={e_mp2_corr!r} e_ccsd={e_ccsd!r} e_ccsd_t={e_ccsd_t!r}")
    print(f"forbidden lines: 1b={n1l} (max {m1l:e})  2b={n2l} (max {m2l:e})")
    print(f"forbidden cells: 1b={n1c} (max {m1c:e})  2b={n2c} (max {m2c:e})")
    print(f"nonzero cells: 1b={rep['n_cells'][0]} 2b={rep['n_cells'][1]}")
    print(f"mo noise: 1b={noise1:e} 2b={noise2:e}")
    print(f"{DUMP}: {os.path.getsize(path)} bytes, {sum(1 for _ in open(path))} lines")


if __name__ == "__main__":
    main()
