import streamlit as st
import pandas as pd
import time
import os
import altair as alt
import numpy as np
import psutil
from pynvml import *

st.set_page_config(page_title="Training Dashboard", layout="wide")
st.title("📈 Real-Time MeshGraphNet Training")

log_file = "train_log.csv"

if not os.path.exists(log_file):
    st.warning("Waiting for training to start...")
    st.stop()

# --- Helper: get min and max values with corresponding steps ---
def get_min_max(df, col):
    if col not in df.columns:
        return None, None, None, None
    min_val, max_val = df[col].min(), df[col].max()
    tolerance = 1e-6
    min_row = df[np.isclose(df[col], min_val, atol=tolerance)]
    max_row = df[np.isclose(df[col], max_val, atol=tolerance)]
    min_step = min_row["step"].values[0] if not min_row.empty else None
    max_step = max_row["step"].values[0] if not max_row.empty else None
    return min_val, min_step, max_val, max_step

# --- Zoomable and filtered velocity loss chart ---
def display_zoomable_loss(df, loss_name):
    if loss_name not in df.columns:
        return

    st.subheader(f"{loss_name.replace('_', ' ').title()} (Filtered 0 ≤ Loss ≤ 8) with Zoom")

    min_val, min_step, max_val, max_step = get_min_max(df, loss_name)
    col1, col2 = st.columns(2)
    col1.metric(f"Min {loss_name}", f"{min_val:.4f}", help=f"At step {min_step}")
    col2.metric(f"Max {loss_name}", f"{max_val:.4f}", help=f"At step {max_step}")

    filtered_df = df[df[loss_name].between(0, 8)]

    chart = alt.Chart(filtered_df).mark_line().encode(
        x=alt.X("step", title="Step"),
        y=alt.Y(loss_name, title=loss_name.replace('_', ' ').title()),
        tooltip=["step", loss_name]
    ).interactive(bind_y=True)

    st.altair_chart(chart, use_container_width=True)

# --- Raw filtered stress loss chart (no zoom) ---
def display_filtered_loss(df, loss_name, color=None):
    if loss_name not in df.columns:
        return

    st.subheader(f"{loss_name.replace('_', ' ').title()}")

    min_val, min_step, max_val, max_step = get_min_max(df, loss_name)
    col1, col2 = st.columns(2)
    col1.metric(f"Min {loss_name}", f"{min_val:.4f}", help=f"At step {min_step}")
    col2.metric(f"Max {loss_name}", f"{max_val:.4f}", help=f"At step {max_step}")

    filtered_df = df[df[loss_name].between(0, 8)]
    chart_base = alt.Chart(filtered_df).mark_line() if color is None else alt.Chart(filtered_df).mark_line(color=color)

    chart = chart_base.encode(
        x="step",
        y=alt.Y(loss_name, title=loss_name.replace('_', ' ').title()),
        tooltip=["step", loss_name]
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# --- Display system stats ---
def display_system_stats():
    st.sidebar.header("💻 System Stats")
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    st.sidebar.metric("CPU Usage", f"{cpu}%")
    st.sidebar.metric("RAM Usage", f"{ram.percent}%", help=f"{ram.used // (1024**2)} MB used of {ram.total // (1024**2)} MB")
    st.sidebar.metric("Disk Usage", f"{disk.percent}%", help=f"{disk.used // (1024**3)} GB used of {disk.total // (1024**3)} GB")

# --- Display GPU 0 and Integrated GPU 1 ---
def display_gpu_stats():
    try:
        nvmlInit()

        # GPU 0 — your dedicated GPU
        handle0 = nvmlDeviceGetHandleByIndex(0)
        mem0 = nvmlDeviceGetMemoryInfo(handle0)
        util0 = nvmlDeviceGetUtilizationRates(handle0)
        name0 = nvmlDeviceGetName(handle0)
        if isinstance(name0, bytes):
            name0 = name0.decode()

        st.sidebar.header(f"🎮 GPU 0: {name0}")
        st.sidebar.metric("Usage", f"{util0.gpu} %")
        st.sidebar.metric("Memory", f"{mem0.used // (1024**2)} MB / {mem0.total // (1024**2)} MB")

        # GPU 1 — integrated GPU
        handle1 = nvmlDeviceGetHandleByIndex(1)
        mem1 = nvmlDeviceGetMemoryInfo(handle1)
        util1 = nvmlDeviceGetUtilizationRates(handle1)
        name1 = nvmlDeviceGetName(handle1)
        if isinstance(name1, bytes):
            name1 = name1.decode()

        st.sidebar.header(f"🖥️ Integrated GPU 1: {name1}")
        st.sidebar.metric("Usage", f"{util1.gpu} %")
        st.sidebar.metric("Memory", f"{mem1.used // (1024**2)} MB / {mem1.total // (1024**2)} MB")

        nvmlShutdown()

    except Exception as e:
        st.sidebar.warning(f"GPU stats unavailable: {e}")

# --- Display training time stats ---
def display_timing_stats(df):
    if "time" not in df.columns or len(df) < 2:
        st.info("Timing information is not available.")
        return

    df["time"] = pd.to_datetime(df["time"], unit='s')
    elapsed = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
    step_count = df["step"].iloc[-1] - df["step"].iloc[0]
    avg_step_time = elapsed / step_count if step_count > 0 else 0

    st.subheader("⏱️ Training Time Stats")
    col1, col2 = st.columns(2)
    col1.metric("Total Training Time", f"{elapsed:.1f} s")
    col2.metric("Avg Time per Step", f"{avg_step_time:.2f} s")

# --- Main live loop ---
while True:
    df = pd.read_csv(log_file)

    # --- Current Step Display ---
    if "step" in df.columns and not df.empty:
        current_step = df["step"].iloc[-1]
        st.metric("🚀 Current Step", current_step)

    display_zoomable_loss(df, "velocity_loss")
    display_filtered_loss(df, "stress_loss", color="orange")

    display_timing_stats(df)
    display_system_stats()
    display_gpu_stats()

    st.markdown("---")
    st.markdown(f"Last updated at: `{time.strftime('%H:%M:%S')}`")

    time.sleep(5)
    st.rerun()
