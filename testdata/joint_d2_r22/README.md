# joint_d2_r22 — a stiff stable joint optimum of the `joint_d2` problem

Same files and conventions as `testdata/joint_d2/`, at per-sector rank 2. The orbital metric at this
point is ill-conditioned (eigenvalues from ~3e-7 to ~6e-3), which makes it a sensitive test of the
joint Davidson solvers against the dense reference.
