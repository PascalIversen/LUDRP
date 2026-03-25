import os

MODELS = ["mcd", "qnn", "pnn", "pnne", "edl", "br", "rf"]

MODEL_COLORS = {
    "mcd":  "#E69F00",
    "qnn":  "#009E73",
    "pnn":  "#56B4E9",
    "pnne": "#D55E00",
    "edl":  "#000000",
    "br":   "#0072B2",
    "rf":   "#CC79A7",
}

LINESTYLES = {
    "mcd":  "--",
    "qnn":  (0, (1, 2, 7, 2)),
    "pnn":  "-.",
    "pnne": ":",
    "edl":  (0, (5, 2, 1, 2)),
    "br":   (0, (2, 1, 1, 1)),
    "rf":   (0, (7, 2)),
}

MODEL_TO_FULL_NAME = {
    "mcd": "MC Dropout",
    "qnn": "Quantile NN",
    "pnn": "Gaussian NN",
    "pnne": "Gaussian Ensemble",
    "edl": "Evidential DL",
    "br": "Bayesian Ridge",
    "rf": "Random Forest",
}

SHIFTS = ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]

PNNE_COLORS = {
    "Total": "#D55E00",
    "Epistemic": "#D55E00",
    "Aleatoric": "#D55E00",
}

PNNE_LINESTYLES = {
    "Total": ":",
    "Aleatoric": (0, (1, 3)),
    "Epistemic": "-",
}

EDL_COLORS = {
    "Total": "#000000",
    "Epistemic": "#000000",
    "Aleatoric": "#000000",
}

EDL_LINESTYLES = {
    "Total": (0, (5, 2, 1, 2)),
    "Aleatoric": (0, (1, 3)),
    "Epistemic": "-",
}

COLD_START = True
COLD_START_SUFFIX = "_cold_" if COLD_START else ""
SUFFIX = "z_norm"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "examples", "data")
BASE_PATH = os.path.join(DATA_DIR, "experiments",
                         "5_fold_cross_validation", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")