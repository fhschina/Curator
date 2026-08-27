# ruff: noqa: BLE001, INP001, S110
"""Compatibility aliases for vLLM/Quack with NVIDIA CUTLASS DSL 4.5."""

try:
    from cutlass import base_dsl, cute
    from cutlass.base_dsl.arch import Arch
    from cutlass.cute import core

    if not hasattr(base_dsl, "Arch"):
        base_dsl.Arch = Arch

    for name in ("ThrMma", "TiledMma", "MmaAtom", "ThrCopy"):
        if not hasattr(core, name) and hasattr(cute, name):
            setattr(core, name, getattr(cute, name))
except Exception:
    # Environments that do not import CUTLASS must remain unaffected.
    pass
