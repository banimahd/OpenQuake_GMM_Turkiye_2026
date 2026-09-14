# -*- coding: utf-8 -*-
"""
Banimahd et al. (2026) Turkiye ANN-based Ground-Motion Model
=============================================================

This module implements the ANN-based, region-specific ground-motion model
for Turkiye by Banimahd et al. (2026) as an OpenQuake GSIM/GMPE.

Reference
---------
Banimahd A, Karimzadeh S, et al. (2026).
"Artificial neural network-based non-parametric ground motion models for
 multiple intensity measures in Turkiye."
Engineering Applications of Artificial Intelligence.

Model overview
--------------
- Trained on strong-motion data from Turkiye.
- ML regressor: ensemble of 10 feed-forward neural networks.
- Inputs (in the order used for training and ONNX):
    1. FD         : focal depth (km)
    2. FM         : fault mechanism (1=Normal, 2=Reverse, 3=StrikeSlip)
    3. Mw         : moment magnitude
    4. RJB        : Joyner-Boore distance (km)
    5. Vs30       : averaged shear-wave velocity in the top 30 m (m/s)

Outputs (25 values, in ln-space):
    - PGA, PSA(T): ln(cm/s^2)
    - PGV       : ln(cm/s)
"""

import os
import csv
import numpy as np
import onnxruntime as ort

from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA

# ---------------------------------------------------------------------
# Choose a suitable IM component, robust across OQ versions
# ---------------------------------------------------------------------
_IMC_CANDIDATES = [
    "GMEAN",
    "GEOMETRIC_MEAN",
    "AVERAGE_HORIZONTAL",
    "HORIZONTAL",
    "RANDOM_HORIZONTAL",
]

for _name in _IMC_CANDIDATES:
    if hasattr(const.IMC, _name):
        _IMC_GMEAN = getattr(const.IMC, _name)
        break
else:
    _IMC_GMEAN = list(const.IMC)[0]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "banimahd_turkiye_2026_data",
)

_ONNX_FILE = os.path.join(_DATA_DIR, "onnx_models", "GMM_Turkiye_2026.onnx")
_STDS_FILE = os.path.join(_DATA_DIR, "stds.csv")


# ---------------------------------------------------------------------
# Load standard deviation tables from stds.csv
# ---------------------------------------------------------------------
_SIGMA_INTRA = {}
_TAU_INTER = {}
_PHI_TOTAL = {}


def _load_stddev_tables():
    if not os.path.exists(_STDS_FILE):
        raise IOError(f"Cannot find stds.csv at {_STDS_FILE}")

    with open(_STDS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["ID"]
            _SIGMA_INTRA[key] = float(row["Sigma"])
            _TAU_INTER[key] = float(row["Tau"])
            _PHI_TOTAL[key] = float(row["Phi"])


_load_stddev_tables()


# ---------------------------------------------------------------------
# ONNX session cache
# ---------------------------------------------------------------------
_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        if not os.path.exists(_ONNX_FILE):
            raise IOError(f"Cannot find ONNX file: {_ONNX_FILE}")
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(_ONNX_FILE, sess_options=opt)
    return _SESSION


__all__ = ["Banimahd2026Turkiye"]


# ---------------------------------------------------------------------
# GSIM class
# ---------------------------------------------------------------------
class Banimahd2026Turkiye(GMPE):
    """
    ANN-based Ground-Motion Model for Turkiye (Banimahd et al., 2026).

    This GSIM wraps an ensemble of 10 feed-forward neural networks exported
    to a single ONNX file. It returns the mean ln(IM) and standard deviations
    (intra-event, inter-event, total) for each requested IMT.
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = _IMC_GMEAN

    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.INTRA_EVENT,
        const.StdDev.INTER_EVENT,
        const.StdDev.TOTAL,
    }

    DEFINED_FOR_REFERENCE_VELOCITY = 760.0

    REQUIRES_RUPTURE_PARAMETERS = {"mag", "hypo_depth", "rake"}
    REQUIRES_DISTANCES = {"rjb"}
    REQUIRES_SITES_PARAMETERS = {"vs30"}

    # Mapping from OpenQuake IMT string to stds.csv ID
    _IMT_TO_KEY = {
        "PGA": "ln(PGA)",
        "PGV": "ln(PGV)",
    }

    # Periods supported by the model (in seconds)
    _PERIODS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Compute mean ln(IM) and standard deviations for the requested IMT.

        Returns
        -------
        mean : np.ndarray, shape (N,)
            Mean logarithm of IM at each site.
        stds : list of np.ndarray
            One array per requested stddev type.
        """
        N = len(sites)

        Mw = np.full(N, rup.mag, dtype=float)
        RJB = np.asarray(dists.rjb, dtype=float)
        Vs30 = np.asarray(sites.vs30, dtype=float)
        FD = np.full(N, rup.hypo_depth, dtype=float)

        # Fault mechanism encoding from rake
        rake = float(rup.rake)
        if rake < -30.0:
            FM = np.full(N, 1.0, dtype=float)   # Normal
        elif rake > 30.0:
            FM = np.full(N, 2.0, dtype=float)   # Reverse
        else:
            FM = np.full(N, 3.0, dtype=float)   # Strike-slip

        # Order: FD, FM, Mw, RJB, VS30
        X = np.column_stack([FD, FM, Mw, RJB, Vs30]).astype(np.float32)

        # IMT identification
        imt_str = str(imt)

        if imt_str == "PGA":
            out_idx = 0
            key = "ln(PGA)"
        elif imt_str == "PGV":
            out_idx = 1
            key = "ln(PGV)"
        elif imt_str.startswith("SA(") and imt_str.endswith(")"):
            period = float(imt_str[3:-1])
            if period not in self._PERIODS:
                raise ValueError(
                    f"Period {period} not supported. "
                    f"Supported periods: {self._PERIODS}"
                )
            out_idx = 7 + self._PERIODS.index(period)
            key = f"ln(PSA={period})"
        else:
            raise ValueError(f"IMT {imt_str} not supported in Banimahd2026Turkiye")

        # Run ONNX inference
        sess = _get_session()
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        out = sess.run([output_name], {input_name: X})[0]
        ln_im_train = out[:, out_idx].astype(float)

        # Unit conversion
        # PGA / SA: cm/s^2 -> g (subtract ln(981))
        # PGV    : stays in cm/s
         ln_im = ln_im_train 


        # Standard deviations from stds.csv
        if key not in _SIGMA_INTRA:
            raise KeyError(f"No stddev entry for IM key '{key}' in stds.csv")

        sigma = _SIGMA_INTRA[key]
        tau = _TAU_INTER[key]
        phi = _PHI_TOTAL[key]

        stds = []
        for s in stddev_types:
            if s == const.StdDev.INTRA_EVENT:
                stds.append(np.full_like(ln_im, sigma))
            elif s == const.StdDev.INTER_EVENT:
                stds.append(np.full_like(ln_im, tau))
            elif s == const.StdDev.TOTAL:
                stds.append(np.full_like(ln_im, phi))
            else:
                raise ValueError(f"StdDev type {s} not supported.")

        return ln_im, stds
