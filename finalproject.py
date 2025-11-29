import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# Map names to scipy.stats distributions and number of parameters (shape parameters)
DISTRIBUTIONS = {
    "Normal (norm)": (stats.norm, 0),
    "Exponential (expon)": (stats.expon, 0),
    "Gamma (gamma)": (stats.gamma, 1),
    "Beta (beta)": (stats.beta, 2),
    "Lognormal (lognorm)": (stats.lognorm, 1),
    "Weibull (weibull_min)": (stats.weibull_min, 1),
    "Uniform (uniform)": (stats.uniform, 0),
    "Laplace (laplace)": (stats.laplace, 0),
    "Chi-square (chi2)": (stats.chi2, 1),
    "Pareto (pareto)": (stats.pareto, 1),
}

st.set_page_config(page_title="Histogram Fitter", layout="wide")

st.title("Histogram Fitting Webapp")

st.markdown(
    "Upload or type data, fit different distributions, and see the histogram with the fitted curve. "
    "You can also manually adjust parameters."
)

# 1. Data input area
st.header("1. Data input")

col_input_left, col_input_right = st.columns(2)

with col_input_left:
    st.subheader("Type or paste data")
    manual_text = st.text_area(
        "Enter numbers separated by spaces, commas, or new lines",
        value="",
        height=150,
    )

    parse_button = st.button("Use typed data")

with col_input_right:
    st.subheader("Upload CSV file")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with one column of numeric data, or pick a column afterwards",
        type=["csv"],
    )

# Function to parse manual text
def parse_manual_data(text: str) -> np.ndarray:
    if not text.strip():
        return np.array([])
    # Replace commas and newlines with spaces, then split
    cleaned = text.replace(",", " ").replace("\n", " ")
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    try:
        data = np.array(list(map(float, parts)))
    except ValueError:
        st.error("Could not parse some values. Please check your input.")
        return np.array([])
    return data

data = np.array([])

# Prefer manually parsed data if button pressed
if parse_button:
    data = parse_manual_data(manual_text)

# If no manual data or not using it, try CSV
if data.size == 0 and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            data = df.iloc[:, 0].dropna().values.astype(float)
            st.success("Using the only column in CSV as data.")
        else:
            # Let the user choose a column
            col_name = st.selectbox(
                "Select a column from the uploaded CSV",
                df.columns,
            )
            data = df[col_name].dropna().values.astype(float)
            st.success(f"Using column '{col_name}' as data.")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

if data.size == 0:
    st.warning("No data loaded yet. Please type some numbers or upload a CSV.")
    st.stop()

# Remove NaNs, infinities, etc
data = data[np.isfinite(data)]
if data.size == 0:
    st.error("All data values are NaN or infinite. Please provide valid numbers.")
    st.stop()

st.write(f"Number of data points: {data.size}")
st.write(f"Example values: {data[:10]}")

# 2. Distribution choice and automatic fitting
st.header("2. Distribution fitting")

dist_name = st.selectbox(
    "Choose a distribution to fit",
    list(DISTRIBUTIONS.keys()),
    index=0,
)

dist, n_shapes = DISTRIBUTIONS[dist_name]

# Choose number of histogram bins
bins = st.slider("Number of histogram bins", min_value=5, max_value=100, value=30)

# Automatic fit
with st.spinner("Fitting distribution to data..."):
    try:
        fitted_params = dist.fit(data)
    except Exception as e:
        st.error(f"Error fitting distribution: {e}")
        st.stop()

st.subheader("Automatic fit parameters")

# Interpret parameters as shape parameters, loc, scale
if n_shapes == 0:
    # (loc, scale)
    loc_auto, scale_auto = fitted_params
    shape_params_auto = []
else:
    shape_params_auto = list(fitted_params[:-2])
    loc_auto = fitted_params[-2]
    scale_auto = fitted_params[-1]

st.write("Shape parameters:", shape_params_auto if shape_params_auto else "None")
st.write("loc:", loc_auto)
st.write("scale:", scale_auto)

# 3. Plot with automatic fit and compute error
st.header("3. Visualization and fit quality")

# Create x grid for PDF
x_min = data.min()
x_max = data.max()
x_range = x_max - x_min
if x_range <= 0:
    x_min -= 1.0
    x_max += 1.0
else:
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range

x_grid = np.linspace(x_min, x_max, 500)

def pdf_from_params(dist_obj, x_vals, shape_params, loc, scale):
    if shape_params:
        return dist_obj.pdf(x_vals, *shape_params, loc=loc, scale=scale)
    else:
        return dist_obj.pdf(x_vals, loc=loc, scale=scale)

# Automatic PDF
pdf_auto = pdf_from_params(dist, x_grid, shape_params_auto, loc_auto, scale_auto)

# Histogram (density normalized)
hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[-1:1:-1])  
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Error metric: mean squared error between hist density at bin centers and pdf
pdf_at_centers_auto = pdf_from_params(dist, bin_centers, shape_params_auto, loc_auto, scale_auto)
mse_auto = np.mean((hist_vals - pdf_at_centers_auto) ** 2)
max_err_auto = np.max(np.abs(hist_vals - pdf_at_centers_auto))

# Plot automatic fit
fig_auto, ax_auto = plt.subplots(figsize=(6, 4))
ax_auto.hist(data, bins=bins, density=True, alpha=0.5, label="Data histogram")
ax_auto.plot(x_grid, pdf_auto, "r-", label="Automatic fit")
ax_auto.set_title(f"{dist_name} fit (automatic)")
ax_auto.set_xlabel("Value")
ax_auto.set_ylabel("Density")
ax_auto.legend()

st.pyplot(fig_auto)

st.markdown(
    f"Automatic fit mean squared error: `{mse_auto:.5f}`, "
    f"maximum absolute error: `{max_err_auto:.5f}`"
)

# 4. Manual fitting with sliders
st.header("4. Manual fitting")

st.markdown(
    "Use the sliders to manually adjust parameters of the current distribution and see how the curve changes."
)

manual_expander = st.expander("Manual parameter controls", expanded=True)

with manual_expander:
    manual_params = []

    # Range guesses from automatic fit, provide reasonable slider ranges
    # Shape parameters
    for i, shp in enumerate(shape_params_auto):
        shp_val = st.slider(
            f"Shape parameter {i+1}",
            min_value=float(shp * 0.1 if shp != 0 else 0.1),
            max_value=float(shp * 3 if shp != 0 else 5.0),
            value=float(shp),
            step=float(abs(shp) * 0.05 if shp != 0 else 0.1),
        )
        manual_params.append(shp_val)

    # loc parameter
    loc_val = st.slider(
        "loc",
        min_value=float(x_min - x_range),
        max_value=float(x_max + x_range),
        value=float(loc_auto),
        step=float(x_range / 100 if x_range > 0 else 0.1),
    )

    # scale parameter
    scale_min = max(scale_auto * 0.1, 1e-3)
    scale_max = scale_auto * 5 if scale_auto > 0 else 10
    scale_val = st.slider(
        "scale",
        min_value=float(scale_min),
        max_value=float(scale_max),
        value=float(scale_auto),
        step=float(scale_auto / 50 if scale_auto > 0 else 0.1),
    )

# Manual PDF and error
pdf_manual = pdf_from_params(dist, x_grid, manual_params, loc_val, scale_val)
pdf_at_centers_manual = pdf_from_params(dist, bin_centers, manual_params, loc_val, scale_val)

mse_manual = np.mean((hist_vals - pdf_at_centers_manual) ** 2)
max_err_manual = np.max(np.abs(hist_vals - pdf_at_centers_manual))

fig_manual, ax_manual = plt.subplots(figsize=(6, 4))
ax_manual.hist(data, bins=bins, density=True, alpha=0.5, label="Data histogram")
ax_manual.plot(x_grid, pdf_manual, "g-", label="Manual fit")
ax_manual.set_title(f"{dist_name} fit (manual)")
ax_manual.set_xlabel("Value")
ax_manual.set_ylabel("Density")
ax_manual.legend()

st.pyplot(fig_manual)

st.markdown(
    f"Manual fit mean squared error: `{mse_manual:.5f}`, "
    f"maximum absolute error: `{max_err_manual:.5f}`"
)

st.header("5. Summary")

st.markdown(
    "This app lets you compare automatic parameter fitting from SciPy with your own manual choices. "
    "You can try different distributions from the drop-down menu and see how well they match your data."
)
