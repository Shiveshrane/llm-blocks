"""Component specifications.

Every building block in ``llm_blocks.nn`` (``GQA``, ``SwiGLU``, ``RMSNorm``, ...) is a
*spec*: a small, frozen, serializable dataclass that describes a component without
allocating any parameters. A spec becomes a real ``torch.nn.Module`` when it is
``build()``-ed with a :class:`BuildContext` that supplies the information the user did
not have to repeat (model width, layer index, depth, ...).

Separating description from construction is what makes the API composable:

* specs can be reused across layers (``[nn.Block(...)] * 12`` builds 12 independent
  blocks with independent parameters),
* configs can be edited with ``replace()`` for ablations,
* configs round-trip through JSON for reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Type, TypeVar

__all__ = ["Spec", "BuildContext", "SpecError", "spec_from_dict", "register_spec"]

T = TypeVar("T", bound="Spec")

_REGISTRY: Dict[str, Type["Spec"]] = {}


class SpecError(ValueError):
    """Raised for invalid or incompatible component configurations."""


def register_spec(cls: Type[T]) -> Type[T]:
    name = cls.__name__
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        existing = _REGISTRY[name]
        if f"{existing.__module__}.{existing.__qualname__}" != f"{cls.__module__}.{cls.__qualname__}":
            raise SpecError(f"Spec name collision: {name} ({existing.__module__} vs {cls.__module__})")
    _REGISTRY[name] = cls
    return cls


@dataclass(frozen=True)
class BuildContext:
    """Information available to a spec when it is turned into a module.

    Attributes:
        dim: width of the residual stream.
        layer_idx: index of the block being built (0-based), or ``None`` for model-level parts.
        num_layers: total number of blocks in the model.
        max_seq_len: maximum sequence length the model is configured for.
        init_std: standard deviation used for weight initialization.
    """

    dim: int
    layer_idx: int | None = None
    num_layers: int = 1
    max_seq_len: int = 2048
    init_std: float = 0.02

    def with_dim(self, dim: int) -> "BuildContext":
        return dataclasses.replace(self, dim=dim)


@dataclass(frozen=True)
class Spec:
    """Base class of every component spec.

    Subclasses are frozen keyword-only dataclasses. They are registered by class name
    so that ``to_dict()`` / ``spec_from_dict()`` can round-trip any configuration.
    """

    kind: ClassVar[str] = "spec"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        register_spec(cls)

    # -- construction -------------------------------------------------------------------
    def build(self, ctx: BuildContext):  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__}.build is not implemented")

    # -- editing ------------------------------------------------------------------------
    def replace(self: T, **changes: Any) -> T:
        """Return a copy with some fields replaced (like ``dataclasses.replace``)."""
        valid = {f.name for f in fields(self)}
        unknown = set(changes) - valid
        if unknown:
            raise SpecError(
                f"{type(self).__name__} has no field(s) {sorted(unknown)}; valid fields: {sorted(valid)}"
            )
        return dataclasses.replace(self, **changes)

    # -- serialization ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):
            out[f.name] = _encode(getattr(self, f.name))
        return out

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        spec = spec_from_dict(data)
        if not isinstance(spec, cls):
            raise SpecError(f"Expected a {cls.__name__}, got {type(spec).__name__}")
        return spec

    @classmethod
    def from_json(cls: Type[T], text: str) -> T:
        return cls.from_dict(json.loads(text))

    # -- display ------------------------------------------------------------------------
    def short_repr(self) -> str:
        """Compact representation showing only non-default fields."""
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            default = f.default if f.default is not dataclasses.MISSING else (
                f.default_factory() if f.default_factory is not dataclasses.MISSING else dataclasses.MISSING
            )
            if default is not dataclasses.MISSING and v == default:
                continue
            parts.append(f"{f.name}={_short(v)}")
        return f"{type(self).__name__}({', '.join(parts)})"


def _short(v: Any) -> str:
    if isinstance(v, Spec):
        return v.short_repr()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_short(x) for x in v) + "]"
    return repr(v)


def _encode(v: Any) -> Any:
    if isinstance(v, Spec):
        return v.to_dict()
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise SpecError(f"Cannot serialize value of type {type(v).__name__}: {v!r}")


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if "type" in v and v["type"] in _REGISTRY:
            return spec_from_dict(v)
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(_decode(x) for x in v)
    return v


def spec_from_dict(data: Dict[str, Any]) -> Spec:
    """Reconstruct any registered spec from its ``to_dict()`` form."""
    if "type" not in data:
        raise SpecError(f"Missing 'type' key in spec dict: {data}")
    name = data["type"]
    if name not in _REGISTRY:
        raise SpecError(f"Unknown spec type {name!r}. Known: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    kwargs = {k: _decode(v) for k, v in data.items() if k != "type"}
    valid = {f.name for f in fields(cls)}
    unknown = set(kwargs) - valid
    if unknown:
        raise SpecError(f"Unknown field(s) {sorted(unknown)} for {name}")
    return cls(**kwargs)
