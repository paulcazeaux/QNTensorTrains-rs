#!/usr/bin/env python
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Generate the SPG6 spin-polarized symmetry-adapted CH2 (3B1) / STO-3G (C2v) FCIDUMP fixture.

Run with the pyscf env, from anywhere:
    ~/miniforge3/envs/pyscf/bin/python testdata/ch2_triplet_c2v/generate.py

Writes `CH2.FCIDUMP` (Molpro-convention ORBSYM + ISYM, 8-fold-packed two-body block, PySCF's own
`tol=1e-15` write threshold) and `meta.txt` (PySCF's OWN reference numbers, so the Rust tests
compare against PySCF rather than against transcribed constants).  NEVER hand-edit either output
— regenerate.  Every self-check below runs BEFORE a single byte is written, the fixture discipline
of `testdata/h2o_c2v/generate.py`, from which `molpro_to_xor`, `assert_integrals_are_symmetry_-
adapted`, `read_dump_entries` and `census` are reused unmodified (they are group-generic).

Why THIS molecule.  The SU(2)
point-group campaign needed an external oracle for `(2S > 0) x irrep` — `h2o_c2v` only covers
`(2S = 0) x identity`.  Triplet methylene at 7 orbitals / 8 electrons has C(7,5)*C(7,3) = 735
determinants in the (5,3) block, so PySCF's FCI is instant, while its two SOMOs (3a1, 1b1) are
NON-degenerate and of DIFFERENT irreps: ROHF reaches the 3B1 configuration by plain aufbau, with
no `irrep_nelec` hand-holding and no degenerate-shell oscillation.  (The obvious |G| = 8
alternative, O2/STO-3G at D2h, has the degenerate pi*g pair as its SOMOs and is version-fragile
for exactly that reason; it is the documented fallback, never a silent substitution.)

This one file pins TWO blocks, which is what makes it worth generating:

  * `e_fci_identity` — the lowest S = 1 state of the TOTALLY SYMMETRIC block, i.e. the block the
    shipped dump's own header selects, because `pyscf.tools.fcidump.write_head` hardcodes
    `ISYM = 1` (§1.6).  A loader that reads ISYM verbatim lands here.
  * `e_fci` — the true 3B1 ground state, reachable only by OVERRIDING the target with
    `gs_irrep_xor`.  This is the campaign's only real-dump, real-oracle NON-IDENTITY-target test.

`molpro_orbsym=True` is mandatory: with it `pyscf.tools.fcidump._convert_orbsym` maps each 0-based
PySCF irrep id through `ORBSYM_MAP[group]` to a 1-based Molpro id, which is what `src/fcidump.rs`
(and `src/pointgroup.rs`'s `MOLPRO_TO_XOR`) expects.  Without it PySCF emits raw 0-based ids and
orbital 0's `ORBSYM` entry is rejected by the reader.
"""

import itertools
import os

import numpy as np
import pyscf
from pyscf import fci, gto, scf, symm
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = "C2v"
UNIT = "Angstrom"
BASIS = "sto-3g"
NORB = 7
NELEC = 8
SPIN = 2  # 2*Sz of the ROHF reference == MS2 of the dump == the campaign's 2S
# Experimental 3B1 geometry (Jensen & Bunker); coordinates are COMPUTED from (r, theta) below and
# then re-derived from the built molecule, never hand-typed as a triple.
R_CH = 1.0753
ANGLE = 133.93
DUMP = "CH2.FCIDUMP"


def geometry(r, theta_deg):
    """C at the origin, both H in the yz plane, C2 axis along z (the h2o_c2v fixture's frame)."""
    half = np.radians(theta_deg) / 2.0
    y, z = r * np.sin(half), r * np.cos(half)
    return f"C 0.0 0.0 0.0; H 0.0 {-y!r} {z!r}; H 0.0 {y!r} {z!r}"


def molpro_to_xor(group):
    """Molpro id - 1 -> XOR index, inverted from PySCF's own `ORBSYM_MAP` (pyscf id -> molpro id).

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
    BEFORE writing, so a fixture can never ship claiming a symmetry it does not have (STOP-12)."""
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
                `TwoBodySymmetry::Eightfold` completion.  This is what the Rust screener walks, so
                it is the number the Rust tests assert against; it is larger than the line count
                (up to 2x for `t`, 8x for `v`) because one written line fills several images.  The
                completion rule below mirrors `read_fcidump` (src/fcidump.rs): explicit entries
                first, then images inserted only into slots not already occupied.
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


def det_census(xor_sites, nup, ndn, wanted):
    """Count determinants of the (nup, ndn) block, and of each irrep sub-block in `wanted`.

    0-based `itertools.combinations(range(NORB), k)`; a determinant's irrep is the XOR of its
    occupied sites' irreps, each spin counted separately (a doubly-occupied site contributes
    `g ^ g = 0`).  Pure counting — iteration order is not load-bearing.
    """
    total = 0
    counts = {g: 0 for g in wanted}
    for up in itertools.combinations(range(NORB), nup):
        gu = 0
        for s in up:
            gu ^= xor_sites[s]
        for dn in itertools.combinations(range(NORB), ndn):
            g = gu
            for s in dn:
                g ^= xor_sites[s]
            total += 1
            if g in counts:
                counts[g] += 1
    return total, counts


def main():
    mol = gto.M(atom=geometry(R_CH, ANGLE), basis=BASIS, unit=UNIT, symmetry=GROUP,
                spin=SPIN, verbose=0)
    # The point group is FORCED, but PySCF assigns it anyway — assert both, so a future PySCF
    # default change cannot silently degrade the fixture to a subgroup.
    assert mol.groupname == GROUP, mol.groupname
    assert mol.topgroup == GROUP, mol.topgroup
    assert mol.nelectron == NELEC and mol.spin == SPIN
    assert mol.nao == NORB, mol.nao
    assert mol.nelec == (5, 3), mol.nelec

    # Geometry sanity, re-derived from the BUILT molecule rather than trusted.
    coords = mol.atom_coords(unit="Angstrom")
    u, v = coords[1] - coords[0], coords[2] - coords[0]
    r_ch = float(np.linalg.norm(u))
    angle = float(np.degrees(np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))))
    assert abs(r_ch - R_CH) < 5e-4, r_ch
    assert abs(angle - ANGLE) < 0.05, angle

    # STOP-1/2: plain aufbau ROHF, no `irrep_nelec` and no level shift.  A reference that needs
    # occupation hand-holding is version-fragile and its meaning drifts silently.
    mf = scf.ROHF(mol).run()
    assert mf.converged, "STOP-1: ROHF did not converge"
    assert list(mf.mo_occ) == [2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0], list(mf.mo_occ)

    orbsym_pyscf = [int(s) for s in mf.mo_coeff.orbsym]  # 0-based XOR ids
    assert len(orbsym_pyscf) == mol.nao == NORB
    orbsym_molpro = [fcidump.ORBSYM_MAP[GROUP][i] for i in orbsym_pyscf]  # 1-based out
    order = len(fcidump.ORBSYM_MAP[GROUP])
    assert set(orbsym_molpro) <= set(range(1, order + 1)), orbsym_molpro
    # STOP-3: an all-1s ORBSYM would be a C1 dump wearing a costume, making every downstream
    # point-group assertion pass trivially.
    assert len(set(orbsym_molpro)) > 1, orbsym_molpro
    irrep_names = [symm.irrep_id2name(GROUP, i) for i in orbsym_pyscf]
    # The two SOMOs must carry DIFFERENT irreps — that is what makes the ground state 3B1 and the
    # non-identity arm of this fixture exist at all.
    assert orbsym_pyscf[3] != orbsym_pyscf[4], irrep_names

    # STOP-12: symmetry self-check on the MO integrals, before writing.
    mo = np.asarray(mf.mo_coeff)
    h1e = mo.T @ np.asarray(mf.get_hcore()) @ mo
    eri = np.asarray(mol.ao2mo(mo, aosym=1)).reshape(NORB, NORB, NORB, NORB)
    xor = molpro_to_xor(GROUP)
    xor_sites = [xor[i - 1] for i in orbsym_molpro]
    # The two spellings of a site's irrep must agree (PySCF ids ARE the XOR indices for these
    # groups); asserting it here means the ORBSYM round trip below tests the FILE, not the map.
    assert xor_sites == orbsym_pyscf, (xor_sites, orbsym_pyscf)
    noise1, noise2 = assert_integrals_are_symmetry_adapted(xor_sites, h1e, eri)

    e_nuc, e_rohf = float(mf.energy_nuc()), float(mf.e_tot)

    # --- PySCF's own physics: the UNCONSTRAINED ground state of the Sz = 1 sector ---
    e_fci, civec = fci.FCI(mf).kernel()
    e_fci = float(e_fci)
    ss, mult = fci.spin_op.spin_square0(civec, NORB, mol.nelec)
    ss, mult = float(ss), float(mult)
    # STOP-4: the Sz = 1 sector also holds S = 2, 3, 4 states; its lowest member is the triplet
    # only if none of them dips below.  Assert, never assume.
    assert abs(ss - 2.0) < 1e-6, f"STOP-4: <S^2> = {ss}, not a triplet"
    assert abs(mult - 3.0) < 1e-6, f"STOP-4: multiplicity {mult}"
    # STOP-5: the whole point of this fixture is a NON-identity target block.
    gs_xor = int(fci.addons.guess_wfnsym(civec, NORB, mol.nelec, np.asarray(orbsym_pyscf)))
    assert gs_xor != 0, "STOP-5: the FCI ground state is totally symmetric"
    gs_molpro = int(fcidump.ORBSYM_MAP[GROUP][gs_xor])
    gs_name = symm.irrep_id2name(GROUP, gs_xor)
    # STOP-11: a (near-)degenerate ground state makes the pinned STATE ambiguous, which would make
    # the purity target of the Rust test ambiguous too.
    two = fci.FCI(mf)
    two.nroots = 2
    roots = [float(x) for x in two.kernel()[0]]
    root_gap = roots[1] - roots[0]
    assert root_gap > 1e-6, f"STOP-11: two lowest roots within {root_gap:e}"

    # STOP-6: `fix_spin_` is a CONTROL, not a crutch.  If spin projection moves the energy, the
    # unprojected pin is ambiguous and this fixture does not ship.
    cis = fci.FCI(mol, mf.mo_coeff)
    fci.addons.fix_spin_(cis, ss=2)
    e_fix = float(cis.kernel()[0])
    fix_agrees = abs(e_fix - e_fci) < 1e-10
    assert fix_agrees, f"STOP-6: fix_spin_ {e_fix!r} vs unprojected {e_fci!r}"

    # --- the IDENTITY block: what the shipped dump's own ISYM = 1 header selects (§1.6) ---
    cid = fci.FCI(mol, mf.mo_coeff)
    cid.wfnsym = 0  # 0-based PySCF/XOR id of the totally symmetric irrep
    fci.addons.fix_spin_(cid, ss=2)
    e_fci_identity, civec_id = cid.kernel()
    e_fci_identity = float(e_fci_identity)
    ss_id, mult_id = fci.spin_op.spin_square0(civec_id, NORB, mol.nelec)
    assert abs(float(ss_id) - 2.0) < 1e-6, f"STOP-4b: identity-block <S^2> = {float(ss_id)}"
    id_xor = int(fci.addons.guess_wfnsym(civec_id, NORB, mol.nelec, np.asarray(orbsym_pyscf)))
    assert id_xor == 0, f"STOP-4b: identity-block state has irrep {id_xor}"
    identity_name = symm.irrep_id2name(GROUP, 0)
    # STOP-7/8: the two blocks must be genuinely distinct, else the two-block test is vacuous.
    block_gap = e_fci_identity - e_fci
    assert block_gap > 1e-4, f"STOP-7/8: block gap {block_gap:e}"

    # --- STOP-9: determinant census of both blocks (0-based sites) ---
    nup, ndn = mol.nelec
    dets_particle, dets = det_census(xor_sites, nup, ndn, (0, gs_xor))
    dets_identity, dets_gs = dets[0], dets[gs_xor]
    assert dets_particle == 735, dets_particle
    assert 0 < dets_identity < dets_particle, (dets_identity, dets_particle)
    assert 0 < dets_gs < dets_particle, (dets_gs, dets_particle)

    # --- write (first output byte) ---
    path = os.path.join(HERE, DUMP)
    fcidump.from_scf(mf, path, tol=1e-15, molpro_orbsym=True)

    # --- post-write round-trip against the SHIPPED bytes ---
    back = fcidump.read(path, verbose=False)
    assert back["NORB"] == NORB and back["NELEC"] == NELEC
    # STOP-10: `from_scf` on an ROHF object writes MS2 = mol.spin; observe it, never hand-edit it.
    assert int(back["MS2"]) == SPIN, back["MS2"]
    assert list(back["ORBSYM"]) == orbsym_molpro, (list(back["ORBSYM"]), orbsym_molpro)
    assert int(back["ISYM"]) == 1, back["ISYM"]  # write_head hardcodes ISYM=1 (§1.6)
    assert abs(float(back["ECORE"]) - e_nuc) < 1e-12
    rep = census(path, GROUP)
    assert rep["orbsym"] == orbsym_molpro and rep["isym"] == 1

    n1l, m1l, n2l, m2l = rep["lines"]
    n1c, m1c, n2c, m2c = rep["cells"]
    with open(os.path.join(HERE, "meta.txt"), "w") as f:
        f.write(f"""# testdata/ch2_triplet_c2v/meta.txt -- generated by generate.py, DO NOT EDIT.
# Keyed `key value...` (never positional: an inserted field would silently corrupt every
# downstream number).  All floats are Python `repr` (shortest round-tripping) of PySCF's OWN
# return values -- nothing here is transcribed or recomputed by hand.
pyscf_version   {pyscf.__version__}
molecule        CH2 (triplet, 3B1)
geometry        {geometry(R_CH, ANGLE)}
r_ch            {R_CH!r}
hch_angle_deg   {ANGLE!r}
unit            {UNIT}
basis           {BASIS}
point_group     {GROUP}
top_group       {mol.topgroup}
norb            {NORB}
nelec           {NELEC}
ms2             {int(back["MS2"])}
isym            {int(back["ISYM"])}
orbsym_molpro   {" ".join(str(s) for s in orbsym_molpro)}
orbsym_pyscf    {" ".join(str(s) for s in orbsym_pyscf)}
irrep_names     {" ".join(irrep_names)}
e_nuc           {e_nuc!r}
e_rohf          {e_rohf!r}
# --- block 1: the UNCONSTRAINED ground state of the Sz = 1 sector.  It is the 3B1 state, so a
# Rust run reaches it only by OVERRIDING the dump's ISYM with `gs_irrep_xor` below.
e_fci           {e_fci!r}
e_fci_wfnsym    {gs_name}
gs_irrep_molpro {gs_molpro}
gs_irrep_xor    {gs_xor}
fci_s_squared   {ss!r}
fci_multiplicity {mult!r}
fix_spin_agrees {"yes" if fix_agrees else "no"}
e_fci_fix_spin  {e_fix!r}
root_gap        {root_gap!r}
# --- block 2: the lowest S = 1 state of the TOTALLY SYMMETRIC block, i.e. the block the shipped
# header selects, because pyscf.tools.fcidump.write_head hardcodes ISYM=1 (a non-identity ISYM
# field cannot come out of PySCF and stays synthetic; see fixture-measurement.md §1.6).
e_fci_identity  {e_fci_identity!r}
identity_irrep_name {identity_name}
identity_s_squared  {float(ss_id)!r}
block_gap       {block_gap!r}
# Determinant counts of the (Nup,Ndn)=({nup},{ndn}) particle block and of its two pinned irrep
# sub-blocks: 0 < sub < particle in both, i.e. the symmetry is genuinely constraining.
dets_particle   {dets_particle}
dets_identity   {dets_identity}
dets_gs         {dets_gs}
# Forbidden-entry census of the SHIPPED bytes, two conventions (see generate.py::census):
#   *_lines  raw file lines (the unique 8-fold triangle PySCF writes);
#   the unsuffixed keys are the distinct t/v tensor CELLS after TwoBodySymmetry::Eightfold
#   completion -- what the Rust screener walks.
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

    print(f"{GROUP}: norb={NORB} nelec={NELEC} ms2={int(back['MS2'])} orbsym={orbsym_molpro}")
    print(f"      names={irrep_names}  somos={irrep_names[3]},{irrep_names[4]}")
    print(f"e_nuc={e_nuc!r} e_rohf={e_rohf!r}")
    print(f"e_fci={e_fci!r} ({gs_name}, xor {gs_xor}, molpro {gs_molpro}) "
          f"<S^2>={ss!r} mult={mult!r}")
    print(f"e_fci_identity={e_fci_identity!r} ({identity_name})  block_gap={block_gap!r}")
    print(f"fix_spin_ control: {e_fix!r} (agrees: {fix_agrees})  root_gap={root_gap!r}")
    print(f"dets: particle={dets_particle} identity={dets_identity} gs={dets_gs}")
    print(f"forbidden lines: 1b={n1l} (max {m1l:e})  2b={n2l} (max {m2l:e})")
    print(f"forbidden cells: 1b={n1c} (max {m1c:e})  2b={n2c} (max {m2c:e})")
    print(f"mo noise: 1b={noise1:e} 2b={noise2:e}")
    print(f"{DUMP}: {os.path.getsize(path)} bytes, {sum(1 for _ in open(path))} lines")


if __name__ == "__main__":
    main()
