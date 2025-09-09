FITTING_SCRIPT_GENERATION_INSTRUCTIONS = """Write a Python script to fit multi-peak luminescence data using lmfit.

REQUIREMENTS:
1. Use lmfit (from lmfit import Model, GaussianModel, LorentzianModel, ConstantModel)
2. Import: matplotlib.pyplot, numpy, json, lmfit, pandas
3. Analyze data for multiple peaks of various shapes
4. Create composite model with multiple Gaussian/Lorentzian components
5. Fit all wells and calculate R² values
6. Re-fit only wells with R² < 0.9
7. Save plot to 'fit_visualization.png'
8. Print results as: FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}, "well_B2": {"R2": 0.87, "peaks": [{"center": 680, "amplitude": 20, "sigma": 10}]}}

EXAMPLE OUTPUT:
FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}}

Write ONLY the Python code:"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS_ERROR = """You are an expert data scientist debugging a Python script. A previously generated script failed to execute. Your task is to analyze the error and provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1.  **Failed Script**: The exact Python code that produced the error.
2.  **Error Message**: The full traceback from the script's execution.

**Your Task:**
1.  Analyze the error message and traceback to identify the bug in the failed script.
2.  Generate a complete, corrected, and executable Python script that fixes the bug while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Failed Script
```python
{failed_script}
```
## Error Message
{error_message}
"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS = """You are an expert data scientist debugging a Python script. A previously generated script executed however, the curve fit was inadequate. Your task is to analyze the old script, fit plot, and the fitted parameters to provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1. **Old Script**: The exact Python code that produced the curve fit.
2. **Curve Fit Plot**: The .png file of the curve fit plot produced by the old script.
3. **Fitted Parameters**: The fitted parameters of the curve fit plot including R2 value as well as the peaks.

**Your Task:**
1. Analyze the old script, curve fit plot, and fitted parameters to identify why the curve fit was inadequate.
2.  Generate a complete, corrected, and executable Python script that fixes the inadequacies while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Old Script
```python
{old_script}
```
## Curve Fit Plot
{old_fit_plot_bytes}

## Fitted Parameters
{old_fitted_parameters}
"""