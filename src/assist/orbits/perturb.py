"""Catalog + perturbation (plan §6.1): the truth orbit is the published element set, perturbed; the catalog output
keeps the published set (mimicking OMM error).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def perturb_omm(
    omm: Mapping[str, Any],
    along_track_s: float = 0.0,
    raan_deg: float = 0.0,
    inc_deg: float = 0.0,
    dn_rev_day: float = 0.0,
) -> dict[str, Any]:
    """A perturbed copy of an OMM dict.

    along_track_s > 0 puts the true satellite that many seconds behind the published one (it arrives later): the
    mean anomaly at epoch moves back by n * along_track_s. raan_deg and inc_deg are added to RAAN and inclination,
    dn_rev_day to the mean motion (so the along-track error grows with time from epoch).
    """
    out = dict(omm)
    n = float(omm["MEAN_MOTION"])  # rev/day
    out["MEAN_ANOMALY"] = (float(omm["MEAN_ANOMALY"]) - n * 360.0 / 86400.0 * along_track_s) % 360.0
    out["RA_OF_ASC_NODE"] = (float(omm["RA_OF_ASC_NODE"]) + raan_deg) % 360.0
    out["INCLINATION"] = float(omm["INCLINATION"]) + inc_deg
    out["MEAN_MOTION"] = n + dn_rev_day
    return out
