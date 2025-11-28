import cv2
import numpy as np
from typing import Final, Dict, Any, List

ARUCO_DICT: Final = cv2.aruco.DICT_4X4_50

MARCADOR_IDS: Final[Dict[int, str]] = {
    1: "Ombro E",
    2: "Ombro D",
    3: "Escapula E",
    4: "Escapula D",
    5: "Pelve E",
    0: "Pelve D"
}
PARES_DE_ANALISE: Final[List[tuple]] = [
    (1, 2, "Ombros", (0, 0, 255)),
    (3, 4, "Escapulas", (0, 0, 255)),
    (5, 0, "Pelve", (0, 0, 255))
]

ARUCO_PARAMS: Final[Dict[str, Any]] = {
    "adaptiveThreshWinSizeMin": 5,
    "adaptiveThreshWinSizeMax": 35,
    "polygonalApproxAccuracyRate": 0.05,
    "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX
}