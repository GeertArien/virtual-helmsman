# Vendored simulator integration

This directory holds the **in-house ship-simulator integration files**. They
are **not** published in this repository and **not** distributed via pip: they
are hand-dropped per machine. `.gitignore` ignores every file in this directory
except this README — please keep it that way.

The `real` simulator backend loads the integration at runtime through the
vendor-neutral `SimulatorWrapper` protocol (see `../wrapper_api.py`). Without
it, the backend raises a `SimulatorError` explaining what is missing; the
`mock` backend (the development default) is unaffected and needs nothing here.

What to drop in here, how to build it, its runtime prerequisites, and the
values for the `simulator.real` config block all come with the integration
notes in the **private** simulator repository — none of that information
belongs in this one.
