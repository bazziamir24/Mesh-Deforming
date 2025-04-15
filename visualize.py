import streamlit as st
import pyvista as pv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio

from time import sleep

st.set_page_config(layout="wide")
st.title("🧠 MeshGraphNet VTK Dashboard – Full Version with Animation & Overlay")

# Load VTK files
vtk_folder = r"C:\Users\Bazzi\Desktop\Mesh Deforming\data\deforming_plate\vtk_files\instance_1"
vtk_files = sorted(glob.glob(os.path.join(vtk_folder, "*.vtk")))

if not vtk_files:
    st.error("❌ No VTK files found.")
    st.stop()

# Session State Setup
if "play" not in st.session_state:
    st.session_state.play = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

# UI - Controls
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("▶️ Play"):
        st.session_state.play = True
    if st.button("⏸️ Pause"):
        st.session_state.play = False
    speed = st.slider("⏱️ Speed (sec/frame)", 0.01, 1.0, 0.2, 0.01)

view = st.selectbox("🔭 View Direction", ["XY", "YZ", "Isometric"])
view_func = {"XY": "view_xy", "YZ": "view_yz", "Isometric": "view_isometric"}[view]

# Frame slider
frame_idx = st.slider("🧭 Frame", 0, len(vtk_files) - 1, st.session_state.frame_idx)
st.session_state.frame_idx = frame_idx
selected_file = vtk_files[frame_idx]

# Load mesh
mesh = pv.read(selected_file)

# Scalar field selection
scalar_options = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
valid_scalars = [s for s in scalar_options if len(mesh.get_array(s)) in [mesh.n_points, mesh.n_cells]]

scalar_name = st.selectbox("🎨 Main Scalar Field", ["None"] + valid_scalars)
overlay_scalar = st.selectbox("🔁 Overlay Scalar Field", ["None"] + valid_scalars)

# Thresholding
if scalar_name != "None":
    scalars = mesh.get_array(scalar_name)
    min_val, max_val = float(np.min(scalars)), float(np.max(scalars))
    th_min, th_max = st.slider("🔎 Scalar Threshold", min_val, max_val, (min_val, max_val))
    mesh = mesh.threshold([th_min, th_max], scalars=scalar_name)

# Slicing
clip_enabled = st.checkbox("✂️ Enable Slicing")
if clip_enabled:
    normal = st.selectbox("Slice Plane Normal", ["x", "y", "z"])
    origin = float(mesh.center[["x", "y", "z"].index(normal)])
    clip_origin = st.slider(f"Slice Position on {normal.upper()}", float(mesh.bounds[0]), float(mesh.bounds[1]), origin)
    mesh = mesh.slice(normal=normal, origin=[clip_origin if i == ["x", "y", "z"].index(normal) else 0 for i in range(3)])

# Rendering
plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
if scalar_name != "None":
    plotter.add_mesh(mesh, scalars=scalar_name, show_edges=True)
else:
    plotter.add_mesh(mesh, show_edges=True)

# Overlay
if overlay_scalar != "None" and overlay_scalar != scalar_name:
    try:
        plotter.add_mesh(mesh.contour(isosurfaces=10, scalars=overlay_scalar), color='orange', line_width=2)
    except Exception as e:
        st.warning(f"Overlay error: {e}")

plotter.add_title(f"Frame {frame_idx}")
getattr(plotter, view_func)()
plotter.add_axes()
img_path = "vtk_render.png"
plotter.screenshot(img_path)
plotter.close()

st.image(img_path, caption=f"🖼️ Frame {frame_idx}")

# Scalar Statistics
if scalar_name != "None":
    st.subheader("📊 Scalar Field Stats")
    st.write(f"🔢 Min: {scalars.min():.4f} | Max: {scalars.max():.4f} | Mean: {scalars.mean():.4f}")
    fig, ax = plt.subplots()
    ax.hist(scalars, bins=30, color='gray')
    ax.set_title(f"{scalar_name} Distribution")
    st.pyplot(fig)

# Export Frame
st.download_button("💾 Download Frame PNG", open(img_path, "rb"), f"frame_{frame_idx}.png")

# Export GIF
with st.expander("🎞️ Export Animation"):
    gif_name = st.text_input("GIF file name", "vtk_animation.gif")
    start = st.number_input("Start frame", 0, len(vtk_files)-1, 0)
    end = st.number_input("End frame", 0, len(vtk_files)-1, min(50, len(vtk_files)-1))

with st.expander("📋 Detailed Scalar Values Per Point / Frame"):
    if scalar_name != "None":
        data_rows = []

        selected_frames = st.slider("Select frame range", 0, len(vtk_files)-1, (0, 10))

        for i in range(selected_frames[0], selected_frames[1] + 1):
            file_path = vtk_files[i]
            mesh = pv.read(file_path)
            frame_name = os.path.basename(file_path)

            points = mesh.points
            scalars_dict = {name: mesh.get_array(name) for name in mesh.array_names}

            for idx, point in enumerate(points):
                row = {
                    "Frame": frame_name,
                    "Point ID": idx,
                    "X": point[0],
                    "Y": point[1],
                    "Z": point[2],
                }

                for scalar_name in scalars_dict:
                    scalar_array = scalars_dict[scalar_name]
                    if len(scalar_array) == len(points):
                        row[scalar_name] = scalar_array[idx]

                data_rows.append(row)

        detailed_df = pd.DataFrame(data_rows)

        # Show in interactive table
        st.dataframe(detailed_df, use_container_width=True)

        # Export CSV
        csv_data = detailed_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Scalar CSV", csv_data, file_name="scalar_values_per_point.csv", mime="text/csv")

    else:
        st.info("Please select a scalar field to display detailed point-level values.")
