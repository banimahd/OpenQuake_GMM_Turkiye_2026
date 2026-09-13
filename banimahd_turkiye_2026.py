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

Standard deviations are currently NOT provided for this GMM.
This is a median-only GMM.
"""

import os
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
    to a single ONNX file. It returns the mean ln(IM) for each requested IMT.

    NOTE: This is a median-only GMM. No standard deviations are provided.
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = _IMC_GMEAN

    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}

    DEFINED_FOR_REFERENCE_VELOCITY = 760.0

    REQUIRES_RUPTURE_PARAMETERS = {"mag", "hypo_depth", "rake"}
    REQUIRES_DISTANCES = {"rjb"}
    REQUIRES_SITES_PARAMETERS = {"vs30"}

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Compute mean ln(IM) for the requested IMT.

        Returns
        -------
        mean : np.ndarray, shape (N,)
            Mean logarithm of IM at each site.
        stds : list of np.ndarray
            Since this is a median-only GMM, we return zeros for TOTAL std.
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
        elif imt_str == "PGV":
            out_idx = 1
        elif imt_str.startswith("SA(") and imt_str.endswith(")"):
            period = float(imt_str[3:-1])
            periods = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                       0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            if period not in periods:
                raise ValueError(
                    f"Period {period} not supported. "
                    f"Supported periods: {periods}"
                )
            out_idx = 7 + periods.index(period)
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
        if imt_str == "PGA" or imt_str.startswith("SA("):
            ln_im = ln_im_train - np.log(981.0)
        else:
            ln_im = ln_im_train

        # Median-only: return zeros for std
        stds = [np.zeros_like(ln_im) for _ in stddev_types]

        return ln_im, stds
