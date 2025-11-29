import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# ---------- CONFIG ----------

st.set_page_config(page_title="Histogram Fitting Webapp", layout="wide")

# Map names to scipy.stats distributions and number of shape parameters
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

# ---------- TITLE ----------

st.title("Histogram Fitting Webapp")

# ---------- 1. DATA INPUT (WITH SUBMIT BUTTON) ----------

# initialize session state storage
if "submitted_data" not in st.session_state:
    st.session_state.submitted_data = None

col_input_left, col_input_right = st.columns(2)

with col_input_left:
    st.subheader("Type or paste data")
    manual_text = st.text_area(
        "Enter numbers separated by spaces, commas, or new lines",
        value="",
        height=150,
        key="manual_text_area"
    )
    submit_manual = st.button("Submit typed data")

with col_input_right:
    st.subheader("Upload CSV file")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with one column of numeric data",
        type=["csv"],
        key="csv_uploader"
    )
    submit_csv = st.button("Use uploaded CSV")

def parse_manual_data(text: str):
    text = text.strip()
    if not text:
        return np.array([])
    cleaned = text.replace(",", " ").replace("\n", " ")
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    try:
        return np.array(list(map(float, parts)))
    except:
        st.error("Could not parse one or more values.")
        return np.array([])

# ---------- HANDLE SUBMIT BUTTONS ----------

# Manual submit
if submit_manual:
    data = parse_manual_data(manual_text)
    if data.size > 0:
        st.session_state.submitted_data = data
        st.success("Manual data submitted successfully!")

# CSV submit
if submit_csv:
    if uploaded_file is None:
        st.error("Upload a CSV first.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 1:
                st.session_state.submitted_data = df.iloc[:,0].dropna().values.astype(float)
            else:
                col_name = st.selectbox("Choose a column", df.columns)
                st.session_state.submitted_data = df[col_name].dropna().values.astype(float)
            st.success("CSV data submitted successfully!")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# ---------- USE SUBMITTED DATA ----------

data = st.session_state.submitted_data

if data is None or data.size == 0:
    st.warning("No valid data submitted yet.")
    st.stop()

data = data[np.isfinite(data)]

st.write(f"Number of data points: {data.size}")


# ---------- FIT DISTRBITUION AND BINS ----------
dist_name = st.selectbox(
    "Choose a distribution to fit",
    list(DISTRIBUTIONS.keys()),
    index=0,
)

dist, n_shapes = DISTRIBUTIONS[dist_name]

bins = st.slider("Number of histogram bins", min_value=5, max_value=100, value=30)

with st.spinner("Fitting distribution to data..."):
    try:
        fitted_params = dist.fit(data)
    except Exception as e:
        st.error(f"Error fitting distribution: {e}")
        st.stop()

if n_shapes == 0:
    # (loc, scale)
    loc_auto, scale_auto = fitted_params
    shape_params_auto = []
else:
    shape_params_auto = list(fitted_params[:-2])
    loc_auto = fitted_params[-2]
    scale_auto = fitted_params[-1]

# ---------- VISUALIZATION & FIT QUALITY ----------

# x-range for PDF grid
x_min = float(data.min())
x_max = float(data.max())
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

hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

pdf_auto = pdf_from_params(dist, x_grid, shape_params_auto, loc_auto, scale_auto)
pdf_at_centers_auto = pdf_from_params(dist, bin_centers, shape_params_auto, loc_auto, scale_auto)
mse_auto = float(np.mean((hist_vals - pdf_at_centers_auto) ** 2))
max_err_auto = float(np.max(np.abs(hist_vals - pdf_at_centers_auto)))

# ---------- ROW 1: PARAMETERS / CONTROLS ----------

param_colA, param_colB = st.columns(2)

with param_colA:
    #st.subheader("Automatic fit parameters")

    st.markdown(
        "**Fitted parameters (automatic):**  \n"
        f"- Shape parameters: `{shape_params_auto if shape_params_auto else 'None'}`  \n"
        f"- loc: `{loc_auto:.4g}`  \n"
        f"- scale: `{scale_auto:.4g}`"
    )

with param_colB:
    #st.subheader("Manual fitting controls")

    manual_params = []

    # Shape parameter sliders
    for i, shp in enumerate(shape_params_auto):
        base = abs(shp) if shp != 0 else 1.0
        shp_min = float(max(shp - 3 * base, 0.001))
        shp_max = float(shp + 3 * base)
        shp_val = st.slider(
            f"Shape parameter {i+1}",
            min_value=shp_min,
            max_value=shp_max,
            value=float(shp),
            step=(shp_max - shp_min) / 100.0,
        )
        manual_params.append(shp_val)

    # loc slider
    loc_val = st.slider(
        "loc",
        min_value=float(x_min - x_range),
        max_value=float(x_max + x_range),
        value=float(loc_auto),
        step=float((x_max - x_min) / 100.0 if x_range > 0 else 0.1),
    )

    # scale slider
    scale_base = scale_auto if scale_auto > 0 else 1.0
    scale_min = float(scale_base * 0.1)
    scale_max = float(scale_base * 5.0)
    scale_val = st.slider(
        "scale",
        min_value=scale_min,
        max_value=scale_max,
        value=float(scale_base),
        step=(scale_max - scale_min) / 100.0,
    )

pdf_manual = pdf_from_params(dist, x_grid, manual_params, loc_val, scale_val)
pdf_at_centers_manual = pdf_from_params(
    dist, bin_centers, manual_params, loc_val, scale_val
)
mse_manual = float(np.mean((hist_vals - pdf_at_centers_manual) ** 2))
max_err_manual = float(np.max(np.abs(hist_vals - pdf_at_centers_manual)))

# ---------- ROW 2: PLOTS ----------

plot_colA, plot_colB = st.columns(2)

with plot_colA:
    st.subheader("Automatic fit")

    fig_auto, ax_auto = plt.subplots(figsize=(4, 3))
    ax_auto.hist(data, bins=bins, density=True, alpha=0.5, label="Histogram")
    ax_auto.plot(x_grid, pdf_auto, "r-", lw=2, label="Automatic PDF")
    ax_auto.set_title(f"{dist_name} — automatic", fontsize=11)
    ax_auto.set_xlabel("Value", fontsize=9)
    ax_auto.set_ylabel("Density", fontsize=9)
    ax_auto.legend(fontsize=8)
    fig_auto.tight_layout()
    st.pyplot(fig_auto, use_container_width=True)

    st.markdown(
        f"**Auto MSE:** `{mse_auto:.5f}`  \n"
        f"**Auto max error:** `{max_err_auto:.5f}`"
    )

with plot_colB:
    st.subheader("Manual fit")

    fig_manual, ax_manual = plt.subplots(figsize=(4, 3))
    ax_manual.hist(data, bins=bins, density=True, alpha=0.5, label="Histogram")
    ax_manual.plot(x_grid, pdf_manual, "g-", lw=2, label="Manual PDF")
    ax_manual.set_title(f"{dist_name} — manual", fontsize=11)
    ax_manual.set_xlabel("Value", fontsize=9)
    ax_manual.set_ylabel("Density", fontsize=9)
    ax_manual.legend(fontsize=8)
    fig_manual.tight_layout()
    st.pyplot(fig_manual, use_container_width=True)

    st.markdown(
        f"**Manual MSE:** `{mse_manual:.5f}`  \n"
        f"**Manual max error:** `{max_err_manual:.5f}`"
    )

