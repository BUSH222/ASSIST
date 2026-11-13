"""Hand-written time ranges: samples/annotations.yaml.

Each entry: `{file, from_s, to_s, label, exclude_from: [...]}`. Times are seconds from the file's start (WAV: sample
0; .raw: the time in its name). `exclude_from` names the analyses that must skip the range: `noise`, `occupancy`,
`humps`, `birdies`, `adc`, `meteor`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

PURPOSES = frozenset({"noise", "occupancy", "humps", "birdies", "adc", "meteor"})


@dataclass(frozen=True)
class Annotation:
    file: str
    from_s: float
    to_s: float
    label: str
    exclude_from: frozenset[str] = field(default_factory=frozenset)

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "from_s": self.from_s,
            "to_s": self.to_s,
            "label": self.label,
            "exclude_from": sorted(self.exclude_from),
        }


def load_annotations(paths: list[str | Path]) -> list[Annotation]:
    out: list[Annotation] = []
    for p in paths:
        entries = yaml.safe_load(Path(p).read_text()) or []
        if not isinstance(entries, list):
            raise ValueError(f"{p}: expected a list of annotations")
        for e in entries:
            excl = frozenset(e.get("exclude_from", []))
            unknown = excl - PURPOSES
            if unknown:
                raise ValueError(f"{p}: unknown exclude_from {sorted(unknown)} (allowed: {sorted(PURPOSES)})")
            a = Annotation(Path(e["file"]).name, float(e["from_s"]), float(e["to_s"]), str(e["label"]), excl)
            if a.to_s <= a.from_s:
                raise ValueError(f"{p}: empty range in {e}")
            out.append(a)
    return out


def excluded_ranges(annotations: list[Annotation], file: str | Path, purpose: str) -> list[tuple[float, float]]:
    """(from_s, to_s) ranges of `file` that `purpose` must skip."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}")
    name = Path(file).name
    return [(a.from_s, a.to_s) for a in annotations if a.file == name and purpose in a.exclude_from]


def keep_mask(t_rel: np.ndarray, ranges: list[tuple[float, float]], pad_s: float = 0.0) -> np.ndarray:
    """True where a time (seconds from file start; a row's start..end must not touch a range) is usable."""
    keep = np.ones(np.shape(t_rel), dtype=bool)
    for a, b in ranges:
        keep &= ~((t_rel >= a - pad_s) & (t_rel <= b + pad_s))
    return keep
