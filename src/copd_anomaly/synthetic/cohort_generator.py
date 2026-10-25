"""
Synthetic COPD Cohort Generator for Multimodal Risk Modeling and Methodological Validation.

MANDATORY NOTICE:
All outputs from this generator are strictly SYNTHETIC and intended solely for testing
multimodal fusion architectures, uncertainty heads, and survival ablations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copd_anomaly.utils.helpers import get_logger

logger = get_logger("synthetic_cohort")

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA BANNER: This cohort is synthetically generated for methodological validation only. "
    "It does not represent real clinical patients or real hospital data."
)


def generate_synthetic_copd_cohort(
    n_patients: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic clinical + imaging biomarker cohort.

    Features:
    - Age, Sex, BMI, Pack-years smoking
    - Spirometry: FEV1_pct_pred, FVC_pct_pred, FEV1_FVC_ratio, GOLD_stage (1..4)
    - Prior exacerbations count
    - Imaging biomarkers: global_anomaly_p90, prm_fsad_pct, prm_emph_pct, upper_lower_ratio
    - Outcomes: 12-month exacerbation event (binary), time_to_exacerbation_months (survival), censored
    """
    rng = np.random.RandomState(seed)

    # 1. Demographics
    ages = rng.normal(loc=65.0, scale=8.0, size=n_patients).clip(45, 88)
    sexes = rng.choice(["M", "F"], size=n_patients, p=[0.55, 0.45])
    bmi = rng.normal(loc=27.0, scale=4.5, size=n_patients).clip(16.0, 42.0)
    pack_years = rng.lognormal(mean=3.5, sigma=0.5, size=n_patients).clip(10, 120)

    # 2. Spirometry (correlated with pack years and age)
    latent_severity = (pack_years / 60.0) + ((ages - 50) / 30.0) + rng.normal(0, 0.5, n_patients)

    fev1_pct = (95.0 - 18.0 * latent_severity + rng.normal(0, 8.0, n_patients)).clip(20.0, 110.0)
    fvc_pct = (98.0 - 12.0 * latent_severity + rng.normal(0, 8.0, n_patients)).clip(35.0, 115.0)
    fev1_fvc = ((fev1_pct / fvc_pct) * 0.75 + rng.normal(0, 0.04, n_patients)).clip(0.20, 0.85)

    # GOLD Staging based on FEV1 % predicted (for FEV1/FVC < 0.70)
    gold_stages = []
    for f1, ratio in zip(fev1_pct, fev1_fvc, strict=True):
        if ratio >= 0.70:
            gold_stages.append(0)  # At risk / Preserved ratio
        elif f1 >= 80.0:
            gold_stages.append(1)  # GOLD 1: Mild
        elif f1 >= 50.0:
            gold_stages.append(2)  # GOLD 2: Moderate
        elif f1 >= 30.0:
            gold_stages.append(3)  # GOLD 3: Severe
        else:
            gold_stages.append(4)  # GOLD 4: Very Severe

    # 3. Clinical history
    prior_exacerbations = rng.poisson(
        lam=(0.5 + 0.4 * np.array(gold_stages)), size=n_patients
    ).clip(0, 8)

    # 4. Quantitative Imaging Features (correlated with GOLD and latent severity)
    prm_emph = (
        1.5 * np.array(gold_stages) ** 2 + rng.exponential(scale=3.0, size=n_patients)
    ).clip(0.1, 45.0)
    prm_fsad = (5.0 * np.array(gold_stages) + rng.normal(8.0, 3.0, size=n_patients)).clip(1.0, 40.0)
    global_anomaly = (
        10.0 + 3.5 * np.array(gold_stages) + 0.3 * prm_emph + rng.normal(0, 2.0, n_patients)
    ).clip(5.0, 40.0)
    upper_lower_ratio = rng.lognormal(mean=0.1, sigma=0.3, size=n_patients)
    heterogeneity_index = rng.gamma(shape=2.0, scale=1.5, size=n_patients)

    # 5. Multimodal Exacerbation Risk Outcome (Ground Truth Data Generating Process)
    # Log-odds of 12-month exacerbation
    log_odds = (
        -3.8
        + 0.45 * prior_exacerbations
        + 0.03 * (100.0 - fev1_pct)
        + 0.08 * global_anomaly
        + 0.04 * prm_fsad
        + 0.03 * prm_emph
        + 0.25 * (upper_lower_ratio > 1.2).astype(float)
        + rng.normal(0, 0.3, n_patients)
    )
    prob_exacerbation = 1.0 / (1.0 + np.exp(-log_odds))
    event_12m = (rng.uniform(0, 1, n_patients) < prob_exacerbation).astype(int)

    # Time to event / survival (months up to 12)
    # Non-events are censored at 12 months; events occur earlier
    time_to_event = np.full(n_patients, 12.0)
    for i in range(n_patients):
        if event_12m[i] == 1:
            # Event occurs between 0.5 and 11.9 months, inversely related to risk
            raw_time = float(rng.exponential(scale=6.0 / (prob_exacerbation[i] + 0.5)))
            time_to_event[i] = round(float(np.clip(raw_time, 0.5, 11.9)), 1)

    df = pd.DataFrame(
        {
            "patient_id": [f"syn_copd_{i:04d}" for i in range(n_patients)],
            "is_synthetic": True,
            "banner": SYNTHETIC_BANNER,
            "age": np.round(ages, 1),
            "sex": sexes,
            "bmi": np.round(bmi, 1),
            "pack_years": np.round(pack_years, 1),
            "fev1_pct_predicted": np.round(fev1_pct, 1),
            "fvc_pct_predicted": np.round(fvc_pct, 1),
            "fev1_fvc_ratio": np.round(fev1_fvc, 3),
            "gold_stage": gold_stages,
            "prior_exacerbations_count": prior_exacerbations,
            "prm_emph_pct": np.round(prm_emph, 2),
            "prm_fsad_pct": np.round(prm_fsad, 2),
            "global_anomaly_score": np.round(global_anomaly, 2),
            "upper_lower_ratio": np.round(upper_lower_ratio, 2),
            "heterogeneity_index": np.round(heterogeneity_index, 2),
            "exacerbation_12m_event": event_12m,
            "time_to_exacerbation_months": time_to_event,
            "censored": (event_12m == 0).astype(int),
        }
    )

    logger.info(
        f"Generated synthetic cohort (N={n_patients}): "
        f"12-month exacerbation rate = {event_12m.mean():.1%}, GOLD 1-4 distribution = {dict(pd.Series(gold_stages).value_counts())}"
    )
    return df
