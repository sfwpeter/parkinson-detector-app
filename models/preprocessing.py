"""Preprocessing helpers for the Streamlit app.

Provides:
- FEATURE_COLUMNS: canonical feature order expected by the model
- validate_input: basic validation of input values
- prepare_features: build dataframe and apply scaler if provided
- get_prediction_explanation: format a simple human explanation
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Define canonical feature order that matches the trained model
FEATURE_COLUMNS: List[str] = [
	'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
	'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
	'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
	'NHR', 'HNR',
	'RPDE', 'DFA', 'D2',
	'spread1', 'spread2', 'PPE'
]


def validate_input(input_data: Dict[str, float]) -> Tuple[List[str], List[str]]:
	"""Simple validation: return (errors, warnings).

	Errors is a list of fatal issues, warnings are non-fatal.
	"""
	errors: List[str] = []
	warnings: List[str] = []

	for feat in FEATURE_COLUMNS:
		if feat not in input_data:
			errors.append(f"Missing feature: {feat}")
			continue
		try:
			v = float(input_data[feat])
		except Exception:
			errors.append(f"Feature {feat} must be numeric")
			continue
		# basic sanity checks - these ranges are generous
		if np.isnan(v):
			errors.append(f"Feature {feat} is NaN")
		if feat in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'] and not (20 < v < 1000):
			warnings.append(f"{feat} has unusual value: {v}")

	return errors, warnings


def prepare_features(input_data: Dict[str, float], scaler: Optional[object]) -> Optional[pd.DataFrame]:
	"""Return a DataFrame with columns in FEATURE_COLUMNS ready for model.predict.

	If scaler is provided, it will be applied (assumed to implement `transform`).
	Returns None on failure.
	"""
	try:
		row = {feat: float(input_data.get(feat, 0.0)) for feat in FEATURE_COLUMNS}
		df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
		if scaler is not None:
			arr = scaler.transform(df.values)
			return pd.DataFrame(arr, columns=FEATURE_COLUMNS)
		return df
	except Exception:
		return None


def get_prediction_explanation(prediction: int, probabilities: List[float]) -> Dict[str, object]:
	"""Return a small explanation dict for display.

	prediction: 0 (negative) or 1 (positive). probabilities: [p0, p1]
	"""
	try:
		conf_pct = float(probabilities[1]) * 100 if len(probabilities) > 1 else float(probabilities[0]) * 100
		status = "Likely Parkinson's" if prediction == 1 else "Unlikely Parkinson's"
		description = (
			"Model indicates a higher probability for Parkinson's-related voice changes. "
			if prediction == 1 else
			"Model indicates a low probability for Parkinson's-related voice changes."
		)

		return {
			'status': status,
			'description': description,
			'confidence': conf_pct
		}
	except Exception:
		return {
			'status': 'Unknown',
			'description': 'Could not compute explanation',
			'confidence': 0.0
		}
