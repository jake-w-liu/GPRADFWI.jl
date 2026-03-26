# API

This page documents the exported public interface of `GPRADFWI.jl`.

## Physical Constants

```@docs
GPRADFWI.c0
GPRADFWI.eps0
GPRADFWI.mu0
GPRADFWI.eta0
```

## Types

```@docs
DebyeMedium
CPMLParams
SourceConfig
FDTDConfig
FWIResult
```

## Configuration And Source Setup

```@docs
create_config
create_source
```

## Forward Modeling

```@docs
run_forward!
run_forward_snapshots
compute_misfit
forward_misfit
```

## Gradient Utilities

```@docs
fd_gradient
ad_gradient
```

## Inversion Drivers

```@docs
run_fwi
run_fwi_multisource
```
