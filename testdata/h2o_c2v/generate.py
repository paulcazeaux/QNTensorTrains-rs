#!/usr/bin/env python
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Generate the PG6 symmetry-adapted H2O / STO-3G (C2v) FCIDUMP fixture.

Run with the pyscf env, from anywhere:
    ~/miniforge3/envs/pyscf/bin/python testdata/h2o_c2v/generate.py

Writes `H2O.FCIDUMP` (Molpro-convention ORBSYM + ISYM, 8-fold-packed two-body block, PySCF's
own `tol=1e-15` write threshold) and `meta.txt` (PySCF's OWN reference numbers, so the Rust
tests compare against PySCF rather than against transcribed constants).  NEVER hand-edit either
output — regenerate.  Every self-check below runs BEFORE a single byte is written (the fixture
discipline of `testdata/pointgroups/generate.py`).

Why this molecule: at 7 orbitals / 10 electrons the (Nup,Ndn)=(5,5) particle block is
C(7,5)^2 = 441 determinants and its totally-symmetric (A1) sub-block is 133, so a determinant
oracle is instant while the symmetry constraint is genuinely non-vacuous — and PySCF's own FCI
is tractable in ~0.03 s, giving the campaign its cross-code physics pin.

`molpro_orbsym=True` is mandatory: with it, `pyscf.tools.fcidump._convert_orbsym` maps each
0-based PySCF irrep id through `ORBSYM_MAP[group]` to a **1-based Molpro id**, which is what
`src/fcidump.rs`'s resolver (and `src/pointgroup.rs`'s `MOLPRO_TO_XOR`, pinned against
`testdata/pointgroups/tables.txt` from this same PySCF module) expects.  Without it PySCF emits
raw 0-based ids and orbital 0's `ORBSYM` entry is rejected by the reader.
"""

import itertools
import os

import numpy as np
import pyscf
from pyscf import fci, gto, scf, symm
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = "C2v"
GEOMETRY = "O 0.0 0.0 0.0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587"
UNIT = "Angstrom"
BASIS = "sto-3g"
NORB = 7
NELEC = 10
DUMP = "H2O.FCIDUMP"


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
    fuse (XOR) to the identity must be numerical noise.  This is checked on the in-memory MO
    integrals BEFORE writing, so a fixture can never ship claiming a symmetry it does not have."""
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
                n_lines=len(ones) + len(twos) + 1)


def main():
    mol = gto.M(atom=GEOMETRY, basis=BASIS, unit=UNIT, symmetry=GROUP, verbose=0)
    # The point group is FORCED, but PySCF assigns it anyway — assert both, so a future PySCF
    # default change cannot silently degrade the fixture to a subgroup.
    assert mol.groupname == GROUP, mol.groupname
    assert mol.topgroup == GROUP, mol.topgroup
    assert mol.nelectron == NELEC and mol.spin == 0
    assert mol.nao == NORB, mol.nao

    # Geometry sanity, computed rather than trusted: r_OH ≈ 0.9579 Å, ∠HOH ≈ 104.5°.
    coords = mol.atom_coords(unit="Angstrom")
    r_oh = float(np.linalg.norm(coords[1] - coords[0]))
    u, v = coords[1] - coords[0], coords[2] - coords[0]
    angle = float(np.degrees(np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))))
    assert abs(r_oh - 0.9579) < 5e-3, r_oh
    assert abs(angle - 104.5) < 0.5, angle

    mf = scf.RHF(mol).run()
    assert mf.converged

    orbsym_pyscf = [int(s) for s in mf.mo_coeff.orbsym]
    assert len(orbsym_pyscf) == mol.nao == NORB
    orbsym_molpro = [fcidump.ORBSYM_MAP[GROUP][i] for i in orbsym_pyscf]
    order = len(fcidump.ORBSYM_MAP[GROUP])
    assert set(orbsym_molpro) <= set(range(1, order + 1)), orbsym_molpro
    # Non-vacuity: an all-1s ORBSYM would be a C1 dump wearing a costume and would make every
    # downstream point-group assertion pass trivially.
    assert len(set(orbsym_molpro)) > 1, orbsym_molpro
    irrep_names = [symm.irrep_id2name(GROUP, i) for i in orbsym_pyscf]

    # Symmetry self-check on the MO integrals, before writing.
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

    # --- PySCF's own physics, pinned into the meta ---
    e_fci = float(fci.FCI(mf).kernel()[0])
    cis = fci.FCI(mol, mf.mo_coeff)
    cis.wfnsym = "A1"
    e_fci_a1 = float(cis.kernel()[0])
    assert abs(e_fci - e_fci_a1) < 1e-10, (e_fci, e_fci_a1)
    civec = fci.FCI(mf).kernel()[1]
    wfnsym = symm.irrep_id2name(
        GROUP, fci.addons.guess_wfnsym(civec, NORB, mol.nelec, np.asarray(orbsym_pyscf)))
    assert wfnsym == "A1", wfnsym  # the FCI ground state IS totally symmetric (ISYM=1's block)

    # Determinant census of the target block, so the Rust oracle's `133` is pinned by PySCF-side
    # combinatorics rather than by a magic number in the test.
    xor_sites = [xor[i - 1] for i in orbsym_molpro]
    nup = ndn = NELEC // 2
    n_particle = n_target = 0
    for up in itertools.combinations(range(NORB), nup):
        for dn in itertools.combinations(range(NORB), ndn):
            n_particle += 1
            g = 0
            for s in up:
                g ^= xor_sites[s]
            for s in dn:
                g ^= xor_sites[s]
            if g == 0:
                n_target += 1
    assert 0 < n_target < n_particle, (n_target, n_particle)

    n1l, m1l, n2l, m2l = rep["lines"]
    n1c, m1c, n2c, m2c = rep["cells"]
    with open(os.path.join(HERE, "meta.txt"), "w") as f:
        f.write(f"""# testdata/h2o_c2v/meta.txt -- generated by generate.py, DO NOT EDIT.
# Keyed `key value...` (NOT the positional `d nelec ms2 e_nuc e_hf e_ref` of the .dat fixtures):
# PG6's fixtures are self-describing FCIDUMPs and a positional file would break on an inserted
# field.  All floats are Python `repr` (shortest round-tripping).
pyscf_version   {pyscf.__version__}
molecule        H2O
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
e_fci           {e_fci!r}
e_fci_wfnsym    {wfnsym}
# Determinant counts of the (Nup,Ndn)=({nup},{ndn}) particle block and of its target-irrep
# (ISYM=1 -> A1) sub-block: 0 < target < particle, i.e. the symmetry is genuinely constraining.
dets_particle   {n_particle}
dets_target     {n_target}
# Forbidden-entry census of the SHIPPED bytes, two conventions (see generate.py::census):
#   *_lines  raw file lines (the unique 8-fold triangle PySCF writes);
#   the unsuffixed keys are the distinct t/v tensor CELLS after TwoBodySymmetry::Eightfold
#   completion -- what tests/pointgroup_pg6.rs's `screen_forbidden` walks.
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
dump_file       {DUMP}
dump_bytes      {os.path.getsize(path)}
dump_lines      {sum(1 for _ in open(path))}
""")

    print(f"{GROUP}: norb={NORB} nelec={NELEC} orbsym={orbsym_molpro} names={irrep_names}")
    print(f"e_nuc={e_nuc!r} e_hf={e_hf!r} e_fci={e_fci!r} (wfnsym {wfnsym})")
    print(f"dets: particle={n_particle} target={n_target}")
    print(f"forbidden lines: 1b={n1l} (max {m1l:e})  2b={n2l} (max {m2l:e})")
    print(f"forbidden cells: 1b={n1c} (max {m1c:e})  2b={n2c} (max {m2c:e})")
    print(f"mo noise: 1b={noise1:e} 2b={noise2:e}")
    print(f"{DUMP}: {os.path.getsize(path)} bytes, {sum(1 for _ in open(path))} lines")


if __name__ == "__main__":
    main()
