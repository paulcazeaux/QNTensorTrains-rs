#!/usr/bin/env python
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Regenerate `testdata/casscf_paper_references.json` — the CASSCF excited-states comparison set.

Run with a pyscf env, from anywhere:

    <pyscf-python> testdata/generate_casscf_references.py

For water / formaldehyde / ethylene in STO-3G and 6-31G this records, per (molecule, basis):
the RHF energy, the full-space FCI roots and gaps where the determinant count allows, the
state-specific ground-state CASSCF energy, the CASCI roots at those orbitals, and the SA-CASSCF
states — alongside the paper's own Table I/II values for comparison.

**The geometries are imported from `generate_fixtures.py`, never restated here.** This file and the
FCIDUMP fixtures must describe the same molecules or the comparison is meaningless, and the
superseded `gen_paper.py` demonstrated exactly that failure mode: it carried its own transcribed
internal coordinates which matched neither the paper's supplementary Table S1 nor its own comment
(r(OH) 0.9686 A / 104.00 deg against the paper's 1.0000 / 104.50, and so on), so every number it
produced was computed at the wrong geometry while being presented as the paper's system.

`NROOTS = 3` is not arbitrary: `notebooks/mknotebooks.py` scrapes the FIRST `"6-31g"` block's
`fci_gaps` array out of this file textually and asserts it has exactly two entries. Three roots
give two gaps. Water must therefore stay the first molecule, and `sto-3g` must stay ahead of
`6-31g` within each `bases` dict, or that scrape silently reads a different system.

Unlike its predecessor this script writes **no integral fixtures** — `generate_fixtures.py` owns
those, in FCIDUMP form.
"""

import json
import math
import os
import sys

import numpy as np
import pyscf
from pyscf import ao2mo, fci, gto, mcscf, scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_fixtures as gf  # noqa: E402  (path set above)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "casscf_paper_references.json")

NROOTS = 3  # -> two gaps; see the module docstring before changing
FCI_DET_CAP = 2_000_000

# Active spaces from the paper: water CAS(8,6), formaldehyde CAS(4,3), ethylene CAS(2,2).
CAS = {"h2o": (8, 6), "hcho": (4, 3), "c2h4": (2, 2)}
PRETTY = {"h2o": "water", "hcho": "formaldehyde", "c2h4": "ethylene"}

# Paper Table I (ground state) and Table II (SS / SA / LR excitations), Hartree.
PAPER = {
    "h2o": dict(gs=-76.039557, ex=[(0.299801, 0.279149, 0.313274),
                                   (0.377632, 0.357262, 0.388773)]),
    "hcho": dict(gs=-113.849431, ex=[(0.144960, 0.147317, 0.138848),
                                     (0.335188, 0.464643, 0.325269)]),
    "c2h4": dict(gs=-78.034921, ex=[(0.384667, 0.388940, 0.354318),
                                    (0.567533, 0.568822, 0.562656)]),
}

# Where each molecule's geometry lives in generate_fixtures.SYSTEMS. Any fixture of that molecule
# will do — the geometry is per-molecule, the basis is varied below.
GEOM_FROM = {"h2o": "h2o_sto3g", "hcho": "hcho_sto3g", "c2h4": "c2h4_631g"}


def ndet(norb, nelec):
    na = nelec // 2
    return math.comb(norb, na) ** 2


def main():
    print(f"pyscf {pyscf.__version__}, numpy {np.__version__}, NROOTS={NROOTS}")
    refs = {}
    for mol_key in ("h2o", "hcho", "c2h4"):  # order is load-bearing (docstring)
        spec = gf.SYSTEMS[GEOM_FROM[mol_key]]
        geom, unit = spec["geometry"], spec["unit"]
        nelecas, ncas = CAS[mol_key]
        rec = {
            "pretty": PRETTY[mol_key],
            "cas": [nelecas, ncas],
            "geometry": geom,
            "unit": unit,
            "geometry_source": spec.get("source", ""),
            "paper": PAPER[mol_key],
            "bases": {},
        }
        print(f"\n===== {PRETTY[mol_key]}  CAS({nelecas},{ncas}) =====", flush=True)

        for basis in ("sto-3g", "6-31g"):  # order is load-bearing (docstring)
            # C1 ON PURPOSE, unlike the FCIDUMP fixtures. The excited roots this file records must
            # span irreps -- the paper's ethylene states include a B1u -- and a symmetry-adapted
            # CASCI/SA-CASSCF solver restricts to a single wavefunction symmetry. Ethylene's
            # CAS(2,2) has only TWO Ag roots, so `nroots = 3` is unsatisfiable under D2h and PySCF
            # raises a broadcast error. Reference energies are physical and symmetry-independent,
            # so dropping the restriction costs nothing and is what makes the comparison valid.
            mol = gto.M(atom=geom, basis=basis, unit=unit, verbose=0)
            mf = scf.RHF(mol).run()
            assert mf.converged, f"{mol_key}/{basis}: RHF did not converge"
            norb, nelec = int(mf.mo_coeff.shape[1]), int(mol.nelectron)
            nd = ndet(norb, nelec)
            b = {"norb": norb, "nelec": nelec, "ndet": nd,
                 "e_hf": float(mf.e_tot), "e_nuc": float(mf.energy_nuc())}
            print(f"  [{basis}] norb={norb} nelec={nelec} ndet={nd:.3e} "
                  f"E_HF={mf.e_tot:.8f}", flush=True)

            # Full-space FCI, where tractable.
            if nd <= FCI_DET_CAP:
                cis = fci.FCI(mf)
                cis.nroots = NROOTS
                roots = [float(e) for e in np.atleast_1d(cis.kernel()[0])]
                b["fci_roots"] = roots
                b["fci_gaps"] = [e - roots[0] for e in roots[1:]]
                print(f"           full FCI gaps = {['%.6f' % g for g in b['fci_gaps']]}",
                      flush=True)
            else:
                b["fci_roots"] = None
                b["fci_gaps"] = None
                print(f"           full FCI skipped (ndet {nd:.2e} > cap)", flush=True)

            # State-specific ground-state CASSCF, then CASCI roots at those orbitals.
            mc = mcscf.CASSCF(mf, ncas, nelecas)
            mc.kernel()
            b["casscf_gs"] = float(mc.e_tot)
            mci = mcscf.CASCI(mf, ncas, nelecas)
            mci.fcisolver.nroots = NROOTS
            mci.kernel(mc.mo_coeff)
            casci = [float(e) for e in np.atleast_1d(mci.e_tot)]
            b["casci_roots"] = casci
            b["casci_gaps"] = [e - casci[0] for e in casci[1:]]

            # SA-CASSCF over the same number of states (the paper's SA column).
            msa = mcscf.CASSCF(mf, ncas, nelecas).state_average_([1.0 / NROOTS] * NROOTS)
            msa.kernel()
            sa = [float(e) for e in np.atleast_1d(msa.e_states)]
            b["sa_states"] = sa
            b["sa_gaps"] = [e - sa[0] for e in sa[1:]]

            delta = b["casscf_gs"] - PAPER[mol_key]["gs"]
            print(f"           CASSCF GS={b['casscf_gs']:.8f} (paper {PAPER[mol_key]['gs']:.6f}, "
                  f"Δ {delta:+.2e})", flush=True)
            print(f"           CASCI gaps={['%.6f' % g for g in b['casci_gaps']]}  "
                  f"SA gaps={['%.6f' % g for g in b['sa_gaps']]}", flush=True)
            rec["bases"][basis] = b
        refs[mol_key] = rec

    with open(OUT, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"\nwrote {OUT}")

    # The scrape `notebooks/mknotebooks.py` performs, replayed here so a schema change fails loudly
    # in the generator rather than silently in a notebook.
    text = open(OUT).read()
    blk = text[text.index('"6-31g"'):]
    span = blk[blk.index('"fci_gaps"'):]
    span = span[span.index("["):span.index("]") + 1]
    n = len(json.loads(span))
    assert n == 2, f"notebook scrape expects 2 fci_gaps in the first 6-31g block, found {n}"
    assert refs["h2o"]["bases"]["6-31g"]["ndet"] == 1656369, "notebook hardcodes this determinant count"
    print("notebook scrape re-verified: 2 fci_gaps, ndet 1656369")


if __name__ == "__main__":
    main()
