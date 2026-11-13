"""Shared training, evaluation and decision logic for the face and voice models.

Both biometric checks answer two questions:

1. **Who is this?** A classifier (Logistic Regression, Random Forest or XGBoost,
   whichever has the lowest cross-validated log loss) gives a probability for
   every enrolled member.
2. **Is it really them, and not someone outside the team?** A classifier always
   names one of the members it knows, even for a stranger. So a sample is
   accepted only when the classifier is confident about the claimed member
   *and* the sample closely resembles that member's enrolled samples (mean
   cosine similarity to their 3 closest enrolled feature vectors).

Both thresholds are chosen on held-out data:

* **genuine attempts**: each photo/recording scored by a model that never saw
  any version of it (leave-one-group-out, groups = source file);
* **teammate impostors**: the same held-out samples claiming another member;
* **outsiders**: each member removed from training entirely and presented as a
  stranger (leave-one-member-out).

The thresholds maximise the average of the genuine acceptance rate, the
teammate rejection rate and the outsider rejection rate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from xgboost import XGBClassifier

SEED = 42
TOP_K = 3  # similarity = mean cosine similarity to the k closest enrolled vectors
PROB_GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)
SIM_GRID = np.round(np.arange(-0.2, 0.96, 0.02), 2)


def candidate_classifiers() -> dict:
    """The three model families compared for each biometric check."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 subsample=0.9, random_state=SEED, n_jobs=1, verbosity=0),
    }


# --------------------------------------------------------------- helpers ---

def _fit(model, X, y_idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clone(model).fit(X, y_idx)


def _proba_full(model, X, present: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities over all ``n_classes``, zero for classes absent from training."""
    out = np.zeros((len(X), n_classes))
    out[:, present] = model.predict_proba(X)
    return out


class Enrolment:
    """Standardised, unit-length feature vectors of every enrolled sample."""

    def __init__(self, X: np.ndarray, y_idx: np.ndarray, n_classes: int):
        self.scaler = StandardScaler().fit(X)
        self.vectors = normalize(self.scaler.transform(X))
        self.labels = y_idx
        self.n_classes = n_classes

    def similarity(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_classes) mean cosine similarity to each member's closest vectors."""
        sims = normalize(self.scaler.transform(X)) @ self.vectors.T
        out = np.full((len(X), self.n_classes), -1.0)
        for k in np.unique(self.labels):
            member = np.sort(sims[:, self.labels == k], axis=1)[:, -TOP_K:]
            out[:, k] = member.mean(axis=1)
        return out


def _relabel(y_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map possibly non-contiguous class indices to 0..k-1 (needed by XGBoost)."""
    present = np.unique(y_idx)
    return present, np.searchsorted(present, y_idx)


def _fit_and_score(model, X_tr, y_tr, X_te, n_classes):
    present, y_local = _relabel(y_tr)
    fitted = _fit(model, X_tr, y_local)
    probs = _proba_full(fitted, X_te, present, n_classes)
    sims = Enrolment(X_tr, y_tr, n_classes).similarity(X_te)
    return probs, sims


# ------------------------------------------------------------ evaluation ---

def cross_validate(model, X, y_idx, groups, n_classes):
    """Leave-one-group-out probabilities and similarities for every row."""
    probs = np.zeros((len(X), n_classes))
    sims = np.zeros((len(X), n_classes))
    for train, test in LeaveOneGroupOut().split(X, y_idx, groups):
        probs[test], sims[test] = _fit_and_score(model, X[train], y_idx[train], X[test], n_classes)
    return probs, sims


def closed_set_metrics(y_idx, probs, classes) -> dict:
    """Accuracy, macro F1 and log loss of the out-of-fold predictions."""
    pred = probs.argmax(axis=1)
    clipped = np.clip(probs, 1e-6, 1)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y_idx, pred)),
        "f1_macro": float(f1_score(y_idx, pred, average="macro")),
        "log_loss": float(log_loss(y_idx, clipped, labels=range(len(classes)))),
    }


def compare_classifiers(X, y_idx, groups, classes) -> pd.DataFrame:
    """Out-of-fold accuracy, F1 and log loss of every candidate model."""
    rows = {}
    for name, model in candidate_classifiers().items():
        probs, _ = cross_validate(model, X, y_idx, groups, len(classes))
        rows[name] = closed_set_metrics(y_idx, probs, classes)
    return pd.DataFrame(rows).T


def verification_attempts(model, X, y_idx, groups, n_classes, outsider_claim: str):
    """Scores (probability, similarity) of genuine, teammate and outsider attempts.

    ``outsider_claim`` is 'argmax' when the outsider is assigned whichever member
    the model finds most likely (the face check), or 'all' when the outsider may
    claim any member (the voice check, where the claim comes from the face).
    """
    probs, sims = cross_validate(model, X, y_idx, groups, n_classes)
    rows = np.arange(len(X))
    genuine = np.c_[probs[rows, y_idx], sims[rows, y_idx]]
    other = np.ones_like(probs, dtype=bool)
    other[rows, y_idx] = False
    teammate = np.c_[probs[other], sims[other]]

    outsider = []
    for k in np.unique(y_idx):
        held = y_idx == k
        if len(np.unique(y_idx[~held])) < 2:
            continue
        p, s = _fit_and_score(model, X[~held], y_idx[~held], X[held], n_classes)
        p[:, k], s[:, k] = -np.inf, -np.inf  # the stranger is not enrolled
        if outsider_claim == "argmax":
            claim = p.argmax(axis=1)
            outsider.append(np.c_[p[np.arange(len(p)), claim], s[np.arange(len(s)), claim]])
        else:
            mask = np.isfinite(p)
            outsider.append(np.c_[p[mask], s[mask]])
    outsider = np.vstack(outsider) if outsider else np.empty((0, 2))
    return {"genuine": genuine, "teammate": teammate, "outsider": outsider}, probs


def _rates(attempts, t_prob, t_sim):
    def accepted(a):
        return (a[:, 0] >= t_prob) & (a[:, 1] >= t_sim)

    return {
        "genuine_accept_rate": float(accepted(attempts["genuine"]).mean()),
        "teammate_reject_rate": float(1 - accepted(attempts["teammate"]).mean()),
        "outsider_reject_rate": float(1 - accepted(attempts["outsider"]).mean())
        if len(attempts["outsider"]) else float("nan"),
    }


def choose_thresholds(attempts) -> tuple[float, float, dict]:
    """Grid-search the probability and similarity thresholds (balanced objective)."""
    best = None
    for t_prob in PROB_GRID:
        for t_sim in SIM_GRID:
            rates = _rates(attempts, t_prob, t_sim)
            values = [v for v in rates.values() if not np.isnan(v)]
            # Balanced score first; then prefer accepting genuine users; then lower thresholds.
            key = (round(np.mean(values), 4), rates["genuine_accept_rate"], -t_prob, -t_sim)
            if best is None or key > best[0]:
                best = (key, float(t_prob), float(t_sim), rates)
    return best[1], best[2], best[3]


# ------------------------------------------------------------- the model ---

@dataclass
class Decision:
    """Outcome of one biometric check."""

    member: str | None
    probability: float
    similarity: float
    accepted: bool
    reason: str
    probabilities: dict = field(default_factory=dict)


@dataclass
class BiometricModel:
    """A trained face or voice model with its acceptance thresholds and metrics."""

    kind: str
    classes: list
    feature_columns: list
    model_name: str
    classifier: object
    enrolment: Enrolment
    prob_threshold: float
    sim_threshold: float
    metrics: dict
    sklearn_version: str = sklearn.__version__

    # -- training --
    @classmethod
    def train(cls, table: pd.DataFrame, feature_columns: list, group_column: str,
              kind: str, outsider_claim: str, verbose: bool = True,
              reuse: "BiometricModel | None" = None) -> "BiometricModel":
        """Compare models, calibrate thresholds, and fit the final model on all rows.

        With ``reuse``, the model family and both thresholds are copied from an
        already calibrated model and only the final fit is done. The demo uses
        this to enrol the team without one member, who then plays a stranger.
        """
        classes = sorted(table["member"].unique())
        X = table[feature_columns].to_numpy(dtype=float)
        y_idx = table["member"].map({m: i for i, m in enumerate(classes)}).to_numpy()
        groups = table[group_column].to_numpy()
        if reuse is not None:
            present, y_local = _relabel(y_idx)
            final = _fit(candidate_classifiers()[reuse.model_name], X, y_local)
            return cls(kind=kind, classes=classes, feature_columns=list(feature_columns),
                       model_name=reuse.model_name, classifier=final,
                       enrolment=Enrolment(X, y_idx, len(classes)),
                       prob_threshold=reuse.prob_threshold, sim_threshold=reuse.sim_threshold,
                       metrics={"note": f"thresholds reused from the full {kind} model"})

        comparison = compare_classifiers(X, y_idx, groups, classes)
        model_name = comparison["log_loss"].idxmin()
        model = candidate_classifiers()[model_name]
        if verbose:
            print(f"[{kind}] cross-validated comparison (leave one {group_column} out):")
            print(comparison.round(3).to_string())
            print(f"[{kind}] selected: {model_name} (lowest log loss)")

        attempts, probs = verification_attempts(model, X, y_idx, groups, len(classes), outsider_claim)
        t_prob, t_sim, rates = choose_thresholds(attempts)
        if "augmentation" in table:  # genuine acceptance on the unmodified samples only
            original = (table["augmentation"] == "original").to_numpy()
            g = attempts["genuine"][original]
            rates["genuine_accept_rate_unmodified"] = float(((g[:, 0] >= t_prob) & (g[:, 1] >= t_sim)).mean())
        cm = confusion_matrix(y_idx, probs.argmax(axis=1), labels=range(len(classes)))
        metrics = {
            "selected_model": model_name,
            "comparison": comparison.to_dict(orient="index"),
            "closed_set": closed_set_metrics(y_idx, probs, classes),
            "confusion_matrix": cm.tolist(),
            "verification": {**rates, "prob_threshold": t_prob, "sim_threshold": t_sim,
                             "n_genuine": len(attempts["genuine"]),
                             "n_teammate": len(attempts["teammate"]),
                             "n_outsider": len(attempts["outsider"])},
            "attempt_scores": {k: np.round(v, 4).tolist() for k, v in attempts.items()},
            "n_rows": len(table), "n_groups": int(len(np.unique(groups))),
            "rows_per_member": table["member"].value_counts().sort_index().to_dict(),
        }
        if verbose:
            v = metrics["verification"]
            print(f"[{kind}] thresholds: probability >= {t_prob:.2f}, similarity >= {t_sim:.2f} | "
                  f"genuine accepted {v['genuine_accept_rate']:.0%}, teammates rejected "
                  f"{v['teammate_reject_rate']:.0%}, outsiders rejected {v['outsider_reject_rate']:.0%}")

        present, y_local = _relabel(y_idx)
        final = _fit(model, X, y_local)
        return cls(kind=kind, classes=classes, feature_columns=list(feature_columns),
                   model_name=model_name, classifier=final,
                   enrolment=Enrolment(X, y_idx, len(classes)),
                   prob_threshold=t_prob, sim_threshold=t_sim, metrics=metrics)

    # -- inference --
    def _scores(self, features: dict) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([[features[c] for c in self.feature_columns]], dtype=float)
        probs = self.classifier.predict_proba(x)[0]
        sims = self.enrolment.similarity(x)[0]
        return probs, sims

    def identify(self, features: dict) -> Decision:
        """Who is this? Accepted only if confident and similar enough (face check)."""
        probs, sims = self._scores(features)
        k = int(probs.argmax())
        return self._decide(k, probs, sims)

    def verify(self, features: dict, claimed: str) -> Decision:
        """Is this the claimed member? (voice check, claim from the face check)."""
        probs, sims = self._scores(features)
        if claimed not in self.classes:
            return Decision(None, 0.0, -1.0, False,
                            f"no {self.kind} sample is enrolled for '{claimed}'",
                            dict(zip(self.classes, probs.round(3))))
        return self._decide(self.classes.index(claimed), probs, sims)

    def _decide(self, k, probs, sims) -> Decision:
        member, p, s = self.classes[k], float(probs[k]), float(sims[k])
        if p < self.prob_threshold:
            reason = f"model confidence for {member} {p:.3f} < {self.prob_threshold:.2f}"
        elif s < self.sim_threshold:
            reason = f"similarity to {member}'s enrolled samples {s:.3f} < {self.sim_threshold:.2f}"
        else:
            reason = f"matches {member} (confidence {p:.2f}, similarity {s:.2f})"
        accepted = p >= self.prob_threshold and s >= self.sim_threshold
        return Decision(member, p, s, accepted, reason, dict(zip(self.classes, probs.round(3))))
