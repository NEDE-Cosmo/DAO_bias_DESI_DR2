# Cobaya DESI DR2 Likelihood with DAO bias effect

Cobaya likelihood `DAO_bias_DESI_DR2` (v3), which implements the DAO bias effect on DESI DR2 data as described in arXiv:2512.15870, based on a simple linear Fisher analysis. The likelihood class introduces 5 nuisance parameters that describe the physics of the DAO bias effect. The likelihood has been tested only with CLASS. The likelihood is based on DESI DR2 data from arXiv:2503.14738.

## Usage sketch

```yaml
theory:
  classy:
    path: /path/to/CLASS
    extra_args:
      output: mPk
      compute_damping_scale: "yes"
    output_params: [z_d, rs_drag]

likelihood:
  DAO_bias_DESI_DR2:
    python_path: /path/to
    mu_npts: 16 #controls the accuracy of the average.
    # dump_theory_to: /path/to/output.txt
```

For the nuisance parameters, choose one of the two physical branches for `Drr` (corresponding to a DAO at smaller or larger scales, respectively):

```yaml
# Negative branch
params:
  Drr:
    prior: {min: -0.6, max: 0.0}

  # relative DAO amplitude
  Atilde:
    prior: {min: 0.0, max: 1}

  # Damping parameters
  hSigma0:
    prior: {min: 3.0, max: 6.0}
  hSigmaD0:
    prior: {min: 3.0, max: 6.0}
  hSigmaDsilk:
    prior: {min: 1.0, max: 15.0}
```

or

```yaml
# Positive branch
params:
  Drr:
    prior: {min: 0.0, max: 0.6}

  # relative DAO amplitude
  Atilde:
    prior: {min: 0.0, max: 1}

  # Damping parameters
  hSigma0:
    prior: {min: 3.0, max: 6.0}
  hSigmaD0:
    prior: {min: 3.0, max: 6.0}
  hSigmaDsilk:
    prior: {min: 1.0, max: 15.0}
```

## Notes

- `compute_damping_scale: yes` is needed so the thermodynamics table contains `r_d`.
- In practice, `z_d` and `rs_drag` should also be listed in the Cobaya `params` block
  as `derived: True`, so they are exposed.

## One-point test

An example one-point Cobaya input is included in:

- `example_evaluate.yaml`
