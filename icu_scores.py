"""
icu_scores.py  -  Deterministic ICU severity scoring engine.

Design rules:
  * No LLM involvement. Every number here is computed from published
    threshold tables so the same inputs always give the same score.
  * Pure Python standard library. No external dependencies.
  * Every function validates its inputs and returns a structured result
    (score + per-item breakdown + a conservative interpretation string)
    so the calling app can show the *why*, not just the number.

Scores are DECISION SUPPORT. They do not replace bedside clinical judgement.

References (thresholds only; no text reproduced):
  qSOFA / SOFA  - Sepsis-3, JAMA 2016; Vincent et al. Intensive Care Med 1996.
  APACHE II     - Knaus et al. Crit Care Med 1985.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreResult:
    name: str
    score: int
    components: dict = field(default_factory=dict)
    interpretation: str = ""
    warnings: list = field(default_factory=list)

    def __str__(self):
        lines = [f"{self.name}: {self.score}"]
        for k, v in self.components.items():
            lines.append(f"   {k}: {v}")
        if self.interpretation:
            lines.append(f"   -> {self.interpretation}")
        for w in self.warnings:
            lines.append(f"   [!] {w}")
        return "\n".join(lines)


def _band(value, table, default=0):
    """table: list of (predicate, points). First match wins."""
    for pred, pts in table:
        if pred(value):
            return pts
    return default


def mean_arterial_pressure(sbp, dbp):
    if sbp is None or dbp is None:
        return None
    return round((sbp + 2 * dbp) / 3)


# --------------------------------------------------------------------------- #
# qSOFA
# --------------------------------------------------------------------------- #
def qsofa(resp_rate, sbp, gcs) -> ScoreResult:
    for label, v, lo, hi in (("resp_rate", resp_rate, 0, 80),
                             ("sbp", sbp, 20, 300),
                             ("gcs", gcs, 3, 15)):
        if v is None or not (lo <= v <= hi):
            raise ValueError(f"qSOFA: {label}={v} is missing or out of range [{lo},{hi}]")

    c = {
        "RR >= 22": 1 if resp_rate >= 22 else 0,
        "Altered mentation (GCS < 15)": 1 if gcs < 15 else 0,
        "SBP <= 100": 1 if sbp <= 100 else 0,
    }
    total = sum(c.values())
    interp = (">= 2: higher risk of poor outcome - escalate assessment, "
              "consider sepsis workup" if total >= 2
              else "< 2: lower risk by qSOFA (does not rule out sepsis)")
    warn = ["qSOFA is a prompt, not a diagnosis. Surviving Sepsis 2021 "
            "advises against using it as a sole screening tool - pair with "
            "clinical judgement / NEWS2 / lactate."]
    return ScoreResult("qSOFA", total, c, interp, warn)


# --------------------------------------------------------------------------- #
# SOFA  (0-24)
# --------------------------------------------------------------------------- #
def sofa(pao2_fio2=None, on_respiratory_support=False,
         platelets=None, bilirubin=None,
         map_mmhg=None, dopamine=0.0, dobutamine=False,
         epinephrine=0.0, norepinephrine=0.0,
         gcs=None, creatinine=None, urine_output_ml_day=None) -> ScoreResult:
    c = {}
    warnings = []

    # Respiration (PaO2/FiO2, mmHg)
    if pao2_fio2 is None:
        c["Respiration"] = 0
        warnings.append("PaO2/FiO2 not provided - respiration scored 0.")
    else:
        if pao2_fio2 < 100 and on_respiratory_support:
            c["Respiration"] = 4
        elif pao2_fio2 < 200 and on_respiratory_support:
            c["Respiration"] = 3
        elif pao2_fio2 < 300:
            c["Respiration"] = 2
        elif pao2_fio2 < 400:
            c["Respiration"] = 1
        else:
            c["Respiration"] = 0

    # Coagulation (platelets x10^3/uL)
    if platelets is None:
        c["Coagulation"] = 0
        warnings.append("Platelets not provided - coagulation scored 0.")
    else:
        c["Coagulation"] = _band(platelets, [
            (lambda x: x < 20, 4), (lambda x: x < 50, 3),
            (lambda x: x < 100, 2), (lambda x: x < 150, 1)])

    # Liver (bilirubin mg/dL)
    if bilirubin is None:
        c["Liver"] = 0
        warnings.append("Bilirubin not provided - liver scored 0.")
    else:
        c["Liver"] = _band(bilirubin, [
            (lambda x: x >= 12.0, 4), (lambda x: x >= 6.0, 3),
            (lambda x: x >= 2.0, 2), (lambda x: x >= 1.2, 1)])

    # Cardiovascular (vasopressors in ug/kg/min for >= 1h)
    if norepinephrine > 0.1 or epinephrine > 0.1 or dopamine > 15:
        c["Cardiovascular"] = 4
    elif (0 < norepinephrine <= 0.1) or (0 < epinephrine <= 0.1) or (5 < dopamine <= 15):
        c["Cardiovascular"] = 3
    elif (0 < dopamine <= 5) or dobutamine:
        c["Cardiovascular"] = 2
    elif map_mmhg is not None and map_mmhg < 70:
        c["Cardiovascular"] = 1
    else:
        c["Cardiovascular"] = 0
        if map_mmhg is None:
            warnings.append("MAP not provided and no vasopressors - CV scored 0.")

    # CNS (GCS)
    if gcs is None:
        c["CNS"] = 0
        warnings.append("GCS not provided - CNS scored 0.")
    else:
        if not (3 <= gcs <= 15):
            raise ValueError(f"SOFA: gcs={gcs} out of range [3,15]")
        c["CNS"] = _band(gcs, [
            (lambda x: x < 6, 4), (lambda x: x <= 9, 3),
            (lambda x: x <= 12, 2), (lambda x: x <= 14, 1)])

    # Renal (creatinine mg/dL, or urine output override)
    renal = 0
    if creatinine is not None:
        renal = _band(creatinine, [
            (lambda x: x >= 5.0, 4), (lambda x: x >= 3.5, 3),
            (lambda x: x >= 2.0, 2), (lambda x: x >= 1.2, 1)])
    elif urine_output_ml_day is None:
        warnings.append("Neither creatinine nor urine output provided - renal scored 0.")
    if urine_output_ml_day is not None:
        if urine_output_ml_day < 200:
            renal = max(renal, 4)
        elif urine_output_ml_day < 500:
            renal = max(renal, 3)
    c["Renal"] = renal

    total = sum(c.values())
    if total >= 2:
        interp = (f"Total {total}. An acute rise of >= 2 in a patient with "
                  "suspected infection meets Sepsis-3 organ dysfunction.")
    else:
        interp = f"Total {total}. Low organ dysfunction burden at this point."
    return ScoreResult("SOFA", total, c, interp, warnings)


# --------------------------------------------------------------------------- #
# APACHE II  (0-71)
# --------------------------------------------------------------------------- #
def apache2(age, temperature_c, map_mmhg, heart_rate, resp_rate,
            fio2, pao2=None, aado2=None, arterial_ph=None,
            sodium=None, potassium=None, creatinine=None,
            acute_renal_failure=False, hematocrit=None, wbc=None, gcs=None,
            chronic_health="none", admission_type="nonoperative") -> ScoreResult:
    """
    admission_type: 'nonoperative', 'emergency_postop', 'elective_postop'
    chronic_health: 'none' or 'severe'  (severe organ insufficiency / immunocompromise)
    Uses worst values in first 24h. pH primary; if pH absent, item is skipped
    with a warning (venous HCO3 alternative not implemented).
    """
    c = {}
    warnings = []

    c["Temperature"] = _band(temperature_c, [
        (lambda x: x >= 41, 4), (lambda x: x >= 39, 3), (lambda x: x >= 38.5, 1),
        (lambda x: x >= 36, 0), (lambda x: x >= 34, 1), (lambda x: x >= 32, 2),
        (lambda x: x >= 30, 3)], default=4)

    c["MAP"] = _band(map_mmhg, [
        (lambda x: x >= 160, 4), (lambda x: x >= 130, 3), (lambda x: x >= 110, 2),
        (lambda x: x >= 70, 0), (lambda x: x >= 50, 2)], default=4)

    c["Heart rate"] = _band(heart_rate, [
        (lambda x: x >= 180, 4), (lambda x: x >= 140, 3), (lambda x: x >= 110, 2),
        (lambda x: x >= 70, 0), (lambda x: x >= 55, 2), (lambda x: x >= 40, 3)],
        default=4)

    c["Resp rate"] = _band(resp_rate, [
        (lambda x: x >= 50, 4), (lambda x: x >= 35, 3), (lambda x: x >= 25, 1),
        (lambda x: x >= 12, 0), (lambda x: x >= 10, 1), (lambda x: x >= 6, 2)],
        default=4)

    # Oxygenation
    if fio2 >= 0.5:
        if aado2 is None:
            raise ValueError("APACHE II: FiO2 >= 0.5 requires A-aDO2.")
        c["Oxygenation (A-aDO2)"] = _band(aado2, [
            (lambda x: x >= 500, 4), (lambda x: x >= 350, 3),
            (lambda x: x >= 200, 2)], default=0)
    else:
        if pao2 is None:
            raise ValueError("APACHE II: FiO2 < 0.5 requires PaO2.")
        c["Oxygenation (PaO2)"] = _band(pao2, [
            (lambda x: x > 70, 0), (lambda x: x > 60, 1),
            (lambda x: x >= 55, 3)], default=4)

    # pH
    if arterial_ph is None:
        c["Arterial pH"] = 0
        warnings.append("Arterial pH not provided - scored 0 (HCO3 alternative not implemented).")
    else:
        c["Arterial pH"] = _band(arterial_ph, [
            (lambda x: x >= 7.7, 4), (lambda x: x >= 7.6, 3), (lambda x: x >= 7.5, 1),
            (lambda x: x >= 7.33, 0), (lambda x: x >= 7.25, 2),
            (lambda x: x >= 7.15, 3)], default=4)

    def _req(name, val):
        if val is None:
            warnings.append(f"{name} not provided - scored 0.")
            return True
        return False

    c["Sodium"] = 0 if _req("Sodium", sodium) else _band(sodium, [
        (lambda x: x >= 180, 4), (lambda x: x >= 160, 3), (lambda x: x >= 155, 2),
        (lambda x: x >= 150, 1), (lambda x: x >= 130, 0), (lambda x: x >= 120, 2),
        (lambda x: x >= 111, 3)], default=4)

    c["Potassium"] = 0 if _req("Potassium", potassium) else _band(potassium, [
        (lambda x: x >= 7, 4), (lambda x: x >= 6, 3), (lambda x: x >= 5.5, 1),
        (lambda x: x >= 3.5, 0), (lambda x: x >= 3, 1), (lambda x: x >= 2.5, 2)],
        default=4)

    if creatinine is None:
        c["Creatinine"] = 0
        warnings.append("Creatinine not provided - scored 0.")
    else:
        base = _band(creatinine, [
            (lambda x: x >= 3.5, 4), (lambda x: x >= 2.0, 3),
            (lambda x: x >= 1.5, 2), (lambda x: x >= 0.6, 0)], default=2)
        c["Creatinine"] = base * 2 if acute_renal_failure else base

    c["Hematocrit"] = 0 if _req("Hematocrit", hematocrit) else _band(hematocrit, [
        (lambda x: x >= 60, 4), (lambda x: x >= 50, 2), (lambda x: x >= 46, 1),
        (lambda x: x >= 30, 0), (lambda x: x >= 20, 2)], default=4)

    c["WBC"] = 0 if _req("WBC", wbc) else _band(wbc, [
        (lambda x: x >= 40, 4), (lambda x: x >= 20, 2), (lambda x: x >= 15, 1),
        (lambda x: x >= 3, 0), (lambda x: x >= 1, 2)], default=4)

    if gcs is None:
        c["GCS points"] = 0
        warnings.append("GCS not provided - GCS points scored 0.")
    else:
        if not (3 <= gcs <= 15):
            raise ValueError(f"APACHE II: gcs={gcs} out of range [3,15]")
        c["GCS points"] = 15 - gcs

    c["Age"] = _band(age, [
        (lambda x: x >= 75, 6), (lambda x: x >= 65, 5), (lambda x: x >= 55, 3),
        (lambda x: x >= 45, 2)], default=0)

    if chronic_health == "severe":
        c["Chronic health"] = 2 if admission_type == "elective_postop" else 5
    else:
        c["Chronic health"] = 0

    total = sum(c.values())
    if total <= 4:
        band = "~4% predicted hospital mortality (nonoperative)"
    elif total <= 9:
        band = "~8%"
    elif total <= 14:
        band = "~15%"
    elif total <= 19:
        band = "~25%"
    elif total <= 24:
        band = "~40%"
    elif total <= 29:
        band = "~55%"
    else:
        band = ">=75%"
    interp = (f"Total {total}. Rough {band}. Mortality varies strongly by "
              "diagnosis; use only for risk communication, not decisions.")
    return ScoreResult("APACHE II", total, c, interp, warnings)


if __name__ == "__main__":
    print("=== TEST 1: qSOFA (expect 2) ===")
    r = qsofa(resp_rate=24, sbp=90, gcs=15); print(r); assert r.score == 2

    print("\n=== TEST 2: SOFA (expect 10) ===")
    r = sofa(pao2_fio2=250, on_respiratory_support=True, platelets=80,
             bilirubin=3.0, map_mmhg=65, gcs=13, creatinine=2.5); print(r)
    assert r.score == 10, r.score

    print("\n=== TEST 3: APACHE II (expect 20) ===")
    r = apache2(age=60, temperature_c=39.0, map_mmhg=65, heart_rate=130,
                resp_rate=30, fio2=0.4, pao2=70, arterial_ph=7.30, sodium=148,
                potassium=5.0, creatinine=2.5, acute_renal_failure=False,
                hematocrit=32, wbc=18, gcs=13); print(r)
    assert r.score == 20, r.score

    print("\n=== TEST 4: input validation (expect ValueError) ===")
    try:
        qsofa(resp_rate=None, sbp=90, gcs=15)
    except ValueError as e:
        print("   correctly raised:", e)

    print("\nAll assertions passed.")
