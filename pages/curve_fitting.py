import os
import tempfile
import json
from io import BytesIO

import pandas as pd
import streamlit as st
import numpy as np
from tools.script import CurveFitting
from tools.memory import MemoryManager
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="wide")
st.title("Curve Fitting")
st.markdown("Upload CSV files, adjust parameters, and generate fitted curves with interactive visualizations.")

if st.button("Clear Cache and Restart Program"):
    st.cache_data.clear()
    st.rerun()

def parse_reads_spec(spec: str, default_max: int) -> list:
    """Parse reads specification string into list of integers"""
    spec = spec.strip().lower()
    if spec == "odd":
        return [i for i in range(1, default_max+1) if i % 2 == 1]
    if spec == "even":
        return [i for i in range(1, default_max+1) if i % 2 == 0]
    if "-" in spec:
        a, b = spec.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        return list(range(min(a, b), max(a, b)+1))
    # comma-separated
    return [int(x.strip()) for x in spec.split(",") if x.strip()]

@st.cache_data
def load_csv_data(file_bytes: bytes, filename: str):
    """Load and cache CSV data"""
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_data
def run_curve_fitting(data_bytes: bytes, comp_bytes: bytes, parameters: dict):
    """Cached curve fitting analysis"""
    try:
        # Write temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as data_temp, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as comp_temp:
            data_temp.write(data_bytes)
            comp_temp.write(comp_bytes)
            data_path = data_temp.name
            comp_path = comp_temp.name

        # Parse parameters
        lum_reads = parse_reads_spec(parameters.get("reads", "1-100"), parameters.get("number_of_reads", 100))
        wells_to_ignore = parameters.get("wells_to_ignore", [])
        
        # Instantiate Curve Fitting Agent
        curve_agent = CurveFitting(
            start_wavelength=parameters.get("start_wavelength", 500),
            end_wavelength=parameters.get("end_wavelength", 850),
            wavelength_step_size=parameters.get("wavelength_step_size", 1),
            time_step=parameters.get("time_step", 100),
            number_of_reads=parameters.get("number_of_reads", 100),
            luminescence_read_numbers=lum_reads,
            wells_to_ignore=wells_to_ignore if isinstance(wells_to_ignore, list) else []
        )

        result = curve_agent.analyze_curve_fitting(data_path=data_path, comp_path=comp_path)
        
        # Cleanup temp files
        try:
            os.unlink(data_path)
            os.unlink(comp_path)
        except:
            pass
            
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

# File Upload Section
st.header("Data Upload")

col1, col2 = st.columns(2)

with col1:
    data_file = st.file_uploader("Upload Data CSV File", type="csv", accept_multiple_files=False, key="data_uploader")
    if data_file:
        # Update path input with uploaded file path
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path.write(data_file.getvalue())
        temp_path.close()
        st.session_state.cf_data_path = temp_path.name
        memory.add_uploaded_file(data_file.name, temp_path.name)
        
        # Load and display data
        data_df = load_csv_data(data_file.getvalue(), data_file.name)
        if data_df is not None and not data_df.empty:
            st.success(f"✅ {data_file.name} uploaded successfully!")
            
            # Display metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric("Rows", len(data_df))
            metric_cols[1].metric("Columns", len(data_df.columns))
            metric_cols[2].metric("Data Points", data_df.size)
            metric_cols[3].metric("Memory", f"{data_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(data_df.head(20), use_container_width=True)
                
                # Basic statistics
                if st.checkbox("Show Statistics", key="show_data_stats"):
                    st.dataframe(data_df.describe(), use_container_width=True)

with col2:
    comp_file = st.file_uploader("Upload Composition CSV File", type="csv", accept_multiple_files=False, key="comp_uploader")
    if comp_file:
        # Update path input with uploaded file path
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path.write(comp_file.getvalue())
        temp_path.close()
        st.session_state.cf_comp_path = temp_path.name
        memory.add_uploaded_file(comp_file.name, temp_path.name)
        
        # Load and display composition data
        comp_df = load_csv_data(comp_file.getvalue(), comp_file.name)
        if comp_df is not None and not comp_df.empty:
            st.success(f"✅ {comp_file.name} uploaded successfully!")
            
            # Display metrics
            metric_cols = st.columns(3)
            metric_cols[0].metric("Rows", len(comp_df))
            metric_cols[1].metric("Columns", len(comp_df.columns))
            metric_cols[2].metric("Data Points", comp_df.size)
            
            # Composition preview
            with st.expander("Composition Preview", expanded=True):
                st.dataframe(comp_df.head(20), use_container_width=True)

# Path Inputs (synced with uploads)
st.header("File Paths")
path_col1, path_col2 = st.columns(2)
with path_col1:
    data_path = st.text_input(
        "Data CSV Path",
        value=st.session_state.get("cf_data_path", ""),
        key="data_path_input",
        help="Path will be updated automatically when you upload a file"
    )
with path_col2:
    comp_path = st.text_input(
        "Composition CSV Path",
        value=st.session_state.get("cf_comp_path", ""),
        key="comp_path_input",
        help="Path will be updated automatically when you upload a file"
    )

# Interactive Data Visualization (if data is uploaded)
if data_file and data_df is not None and not data_df.empty:
    st.header("Interactive Data Visualization")
    
    viz_col1, viz_col2 = st.columns([1, 2])
    
    with viz_col1:
        x_col = st.selectbox("X Axis Column", data_df.columns.tolist(), index=0, key="x_axis_select")
        y_col = st.selectbox("Y Axis Column", data_df.columns.tolist(), index=min(1, len(data_df.columns)-1), key="y_axis_select")
        chart_type = st.selectbox("Chart Type", ["Line", "Scatter", "Bar"], key="chart_type_select")
    
    with viz_col2:
        # Create interactive chart using plotly
        try:
            if chart_type == "Line":
                fig = px.line(data_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif chart_type == "Scatter":
                fig = px.scatter(data_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            else:  # Bar
                fig = px.bar(data_df.head(50), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download chart button
            chart_bytes = fig.to_image(format="png")
            st.download_button(
                "Download Chart (PNG)",
                chart_bytes,
                f"chart_{x_col}_{y_col}.png",
                "image/png",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data_df[x_col], data_df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}")
            ax.grid(True)
            st.pyplot(fig)
            
            # Download matplotlib chart
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                "Download Chart (PNG)",
                buf.getvalue(),
                f"chart_{x_col}_{y_col}.png",
                "image/png",
                use_container_width=True
            )
            plt.close(fig)

# Parameters Section
st.header("Curve Fitting Parameters")
with st.expander("Parameters", expanded=True):
    with st.form("curve_fitting_parameters"):
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            start_wavelength = st.number_input("Wavelength Start (nm)", value=500, step=1, min_value=100, max_value=2000)
            end_wavelength = st.number_input("Wavelength End (nm)", value=850, step=1, min_value=100, max_value=2000)
            wavelength_step_size = st.number_input("Wavelength Step Size", value=1, step=1, min_value=1)
            time_step = st.number_input("Time Step", value=100, step=1, min_value=1)
        
        with param_col2:
            number_of_reads = st.number_input("Number of Reads", value=100, step=1, min_value=1)
            reads = st.text_input("Reads Specification", value="1-100", 
                                 placeholder="e.g. '1-100', 'odd', 'even', or '1,5,7'")
            wells_to_ignore = st.text_input("Wells to Ignore", 
                                           placeholder="comma-separated well ids, e.g. A1,B2",
                                           value="")
        
        submitted = st.form_submit_button("Generate Curve Fit", type="primary", use_container_width=True)

# Curve Fitting Execution
if submitted:
    if not data_path or not comp_path:
        st.error("Please provide both data and composition file paths or upload files.")
        st.stop()
    
    if not os.path.exists(data_path) or not os.path.exists(comp_path):
        st.error("One or both file paths do not exist. Please upload files or check paths.")
        st.stop()
    
    # Read file bytes for processing
    try:
        with open(data_path, 'rb') as f:
            data_bytes = f.read()
        with open(comp_path, 'rb') as f:
            comp_bytes = f.read()
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()
    
    params = {
        "start_wavelength": start_wavelength,
        "end_wavelength": end_wavelength,
        "wavelength_step_size": wavelength_step_size,
        "time_step": time_step,
        "number_of_reads": number_of_reads,
        "reads": reads,
        "wells_to_ignore": [w.strip() for w in wells_to_ignore.split(",") if w.strip()] if wells_to_ignore else []
    }
    
    with st.spinner("Running curve fitting analysis..."):
        result = run_curve_fitting(data_bytes, comp_bytes, params)
    
    if result.get("status") != "success":
        st.error(result.get("message", "Curve fitting failed."))
        st.stop()
    
    st.success("✅ Curve fitting completed successfully!")
    
    # Display Results
    st.header("Fitting Results")
    
    # Display images
    if result.get("analysis_images"):
        img_cols = st.columns(len(result["analysis_images"]))
        for i, img_info in enumerate(result["analysis_images"]):
            with img_cols[i]:
                st.subheader(img_info.get("label", f"Image {i+1}"))
                st.image(img_info.get("data"))
                
                # Download image button
                st.download_button(
                    f"Download {img_info.get('label', 'Image')}",
                    img_info.get("data"),
                    f"{img_info.get('label', 'image').replace(' ', '_').lower()}.png",
                    "image/png",
                    use_container_width=True,
                    key=f"download_img_{i}"
                )
    
    # Display fitting parameters
    fitting_params = result.get("fitting_parameters", {})
    if fitting_params:
        st.subheader("Fitted Parameters")
        
        # Display as metrics
        if "main_well" in fitting_params:
            main_well = fitting_params["main_well"]
            r2_score = main_well.get("R2", 0)
            peaks = main_well.get("peaks", [])
            
            param_cols = st.columns(4)
            param_cols[0].metric("R² Score", f"{r2_score:.4f}")
            param_cols[1].metric("Number of Peaks", len(peaks))
            
            if peaks:
                # Peak parameters table
                peak_data = []
                for i, peak in enumerate(peaks):
                    peak_data.append({
                        "Peak": i+1,
                        "Center (nm)": f"{peak.get('center', 0):.2f}",
                        "Amplitude": f"{peak.get('amplitude', 0):.2f}",
                        "Sigma": f"{peak.get('sigma', 0):.2f}"
                    })
                
                peak_df = pd.DataFrame(peak_data)
                st.dataframe(peak_df, use_container_width=True)
                
                # Download peak parameters
                csv_data = peak_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Peak Parameters (CSV)",
                    csv_data,
                    "peak_parameters.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        # Full JSON view
        with st.expander("Full Fitting Parameters (JSON)", expanded=False):
            st.json(fitting_params)
            
            # Download JSON
            json_data = json.dumps(fitting_params, indent=2).encode('utf-8')
            st.download_button(
                "Download Parameters (JSON)",
                json_data,
                "fitting_parameters.json",
                "application/json",
                use_container_width=True
            )

# Display uploaded files summary
if st.session_state.get("uploaded_files"):
    st.divider()
    st.subheader("Uploaded Files Summary")
    uploaded_summary = pd.DataFrame(st.session_state.uploaded_files)
    st.dataframe(uploaded_summary[["name", "timestamp"]], use_container_width=True, hide_index=True)
