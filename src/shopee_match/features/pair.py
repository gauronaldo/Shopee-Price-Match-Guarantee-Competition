"""Interpretable pair features and a calibrated classical match model."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.features.text import normalize_title

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "tfidf_similarity",
    "phash_similarity",
    "orb_similarity",
    "token_jaccard",
    "exact_normalized_title",
    "digit_jaccard",
    "digit_conflict",
    "quantity_overlap",
    "quantity_conflict",
    "model_token_jaccard",
    "title_length_ratio",
)

_NUMBER = r"\d+(?:[.,]\d+)?"
_QUANTITY_AFTER = re.compile(
    rf"(?<!\w)(?P<value>{_NUMBER})\s*(?P<unit>kg|grams?|gr|mg|liters?|litres?|ml|cl|l|"
    r"tb|gb|mb|pcs?|pieces?|packs?|sets?|cm|mm)(?!\w)",
    flags=re.IGNORECASE,
)
_QUANTITY_BEFORE = re.compile(
    rf"(?<!\w)(?P<unit>packs?|sets?|pcs?|pieces?)\s*(?:of\s*)?(?P<value>{_NUMBER})(?!\w)",
    flags=re.IGNORECASE,
)
_DIGIT = re.compile(_NUMBER)
_MODEL_TOKEN = re.compile(r"(?=\w*[a-z])(?=\w*\d)\w+", flags=re.IGNORECASE)


def _canonical_quantity(value: str, unit: str) -> tuple[str, float]:
    amount = float(value.replace(",", "."))
    normalized_unit = unit.casefold()
    conversions = {
        "kg": ("mass_mg", 1_000_000.0),
        "gram": ("mass_mg", 1_000.0),
        "grams": ("mass_mg", 1_000.0),
        "gr": ("mass_mg", 1_000.0),
        "mg": ("mass_mg", 1.0),
        "liter": ("volume_ml", 1_000.0),
        "liters": ("volume_ml", 1_000.0),
        "litre": ("volume_ml", 1_000.0),
        "litres": ("volume_ml", 1_000.0),
        "l": ("volume_ml", 1_000.0),
        "cl": ("volume_ml", 10.0),
        "ml": ("volume_ml", 1.0),
        "tb": ("storage_mb", 1_048_576.0),
        "gb": ("storage_mb", 1_024.0),
        "mb": ("storage_mb", 1.0),
        "pc": ("count", 1.0),
        "pcs": ("count", 1.0),
        "piece": ("count", 1.0),
        "pieces": ("count", 1.0),
        "pack": ("count", 1.0),
        "packs": ("count", 1.0),
        "set": ("count", 1.0),
        "sets": ("count", 1.0),
        "cm": ("length_mm", 10.0),
        "mm": ("length_mm", 1.0),
    }
    dimension, scale = conversions[normalized_unit]
    return dimension, round(amount * scale, 6)


def extract_quantities(title: str) -> frozenset[tuple[str, float]]:
    """Extract comparable quantities while normalizing common marketplace units."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    matches = [
        *list(_QUANTITY_AFTER.finditer(normalized)),
        *list(_QUANTITY_BEFORE.finditer(normalized)),
    ]
    return frozenset(
        _canonical_quantity(match.group("value"), match.group("unit")) for match in matches
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def pair_feature_values(
    left: CorpusItem,
    right: CorpusItem,
    tfidf_similarity: float,
    phash_similarity: float,
    orb_similarity: float,
) -> npt.NDArray[np.float64]:
    """Create label-blind symmetric features for one candidate pair."""
    left_title = normalize_title(left.title)
    right_title = normalize_title(right.title)
    left_tokens = set(left_title.split())
    right_tokens = set(right_title.split())
    left_digits = set(_DIGIT.findall(left_title))
    right_digits = set(_DIGIT.findall(right_title))
    left_quantities = extract_quantities(left_title)
    right_quantities = extract_quantities(right_title)
    left_models = set(_MODEL_TOKEN.findall(left_title))
    right_models = set(_MODEL_TOKEN.findall(right_title))
    max_length = max(len(left_title), len(right_title), 1)
    quantity_union = left_quantities | right_quantities
    quantity_overlap = (
        len(left_quantities & right_quantities) / len(quantity_union) if quantity_union else 0.0
    )
    return np.asarray(
        [
            tfidf_similarity,
            phash_similarity,
            orb_similarity,
            _jaccard(left_tokens, right_tokens),
            float(left_title == right_title),
            _jaccard(left_digits, right_digits),
            float(bool(left_digits and right_digits and left_digits != right_digits)),
            quantity_overlap,
            float(
                bool(
                    left_quantities
                    and right_quantities
                    and left_quantities.isdisjoint(right_quantities)
                )
            ),
            _jaccard(left_models, right_models),
            min(len(left_title), len(right_title)) / max_length,
        ],
        dtype=np.float64,
    )


@dataclass(slots=True)
class ClassicalPairModel:
    """Standardized Logistic Regression with serializable model diagnostics."""

    scaler: StandardScaler
    classifier: LogisticRegression

    @classmethod
    def fit(
        cls,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
        regularization_c: float,
        seed: int,
    ) -> ClassicalPairModel:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Pair feature matrix has an unexpected shape")
        if labels.shape != (features.shape[0],) or set(labels.tolist()) != {0, 1}:
            raise ValueError("Pair training labels must contain both binary classes")
        scaler = StandardScaler()
        standardized = scaler.fit_transform(features)
        classifier = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(standardized, labels)
        return cls(scaler, classifier)

    def predict_scores(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilities = self.classifier.predict_proba(self.scaler.transform(features))[:, 1]
        return np.asarray(probabilities, dtype=np.float64)

    def diagnostics(self) -> dict[str, object]:
        coefficients = self.classifier.coef_[0]
        return {
            "feature_names": list(FEATURE_NAMES),
            "standardized_coefficients": {
                name: float(value) for name, value in zip(FEATURE_NAMES, coefficients, strict=True)
            },
            "intercept": float(self.classifier.intercept_[0]),
            "scaler_mean": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.scaler.mean_, strict=True)
            },
            "scaler_scale": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.scaler.scale_, strict=True)
            },
        }
