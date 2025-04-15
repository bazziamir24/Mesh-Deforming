# import tensorflow as tf
# import numpy as np
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk
# import os

# # Define paths for the train dataset (ground truth)
# TFRECORD_FILE = "C:/Users/Bazzi/Desktop/Mesh/deepmind-research/meshgraphnets/data/deforming_plate/train.tfrecord"
# OUTPUT_VTK_DIR = "C:/Users/Bazzi/Desktop/Mesh/deepmind-research/meshgraphnets/data/deforming_plate/vtk_outputs/"

# # Ensure output directory exists
# os.makedirs(OUTPUT_VTK_DIR, exist_ok=True)

# def parse_example(example_proto):
#     """Parses a single TFRecord example."""
#     feature_description = {
#         'world_pos': tf.io.FixedLenFeature([], tf.string),  # Node positions
#         'mesh_pos': tf.io.FixedLenFeature([], tf.string),  # Mesh positions
#         'cells': tf.io.FixedLenFeature([], tf.string),  # Mesh connectivity
#         'node_type': tf.io.FixedLenFeature([], tf.string),  # Node types
#         'stress': tf.io.FixedLenFeature([], tf.string),  # Stress values
#     }
#     parsed_features = tf.io.parse_single_example(example_proto, feature_description)

#     # Decode bytes to float/int arrays
#     world_pos = tf.io.decode_raw(parsed_features['world_pos'], tf.float32).numpy()
#     mesh_pos = tf.io.decode_raw(parsed_features['mesh_pos'], tf.float32).numpy()
#     mesh_cells = tf.io.decode_raw(parsed_features['cells'], tf.int32).numpy()
#     node_type = tf.io.decode_raw(parsed_features['node_type'], tf.int32).numpy()
#     stress = tf.io.decode_raw(parsed_features['stress'], tf.float32).numpy()

#     try:
#         world_pos = world_pos.reshape(-1, 3)  # (N*3,) -> (N, 3)
#         mesh_pos = mesh_pos.reshape(-1, 3)  # (N*3,) -> (N, 3)
#         mesh_cells = mesh_cells.reshape(-1, 4)  # (M*4,) -> (M, 4)

#         # Ensure node_type and stress have one value per node
#         expected_nodes = world_pos.shape[0]
#         node_type = node_type.reshape(-1)
#         stress = stress.reshape(-1)

#         if node_type.shape[0] != expected_nodes:
#             print(f"⚠️ Warning: node_type has {node_type.shape[0]} values, expected {expected_nodes}. Padding...")
#             node_type = np.pad(node_type, (0, expected_nodes - node_type.shape[0]), mode='constant', constant_values=-1)

#         if stress.shape[0] != expected_nodes:
#             print(f"⚠️ Warning: stress has {stress.shape[0]} values, expected {expected_nodes}. Padding...")
#             stress = np.pad(stress, (0, expected_nodes - stress.shape[0]), mode='constant', constant_values=0.0)

#     except Exception as e:
#         print(f"❌ ERROR in reshaping: {e}")
#         return None, None, None, None, None
    
#     return world_pos, mesh_pos, mesh_cells, node_type, stress

# # Read all shapes from Train TFRecord dataset
# dataset = tf.data.TFRecordDataset(TFRECORD_FILE)

# try:
#     for idx, raw_record in enumerate(dataset):
#         world_pos, mesh_pos, mesh_cells, node_type, stress = parse_example(raw_record)

#         if world_pos is None or mesh_cells is None:
#             print(f"⚠️ Skipping shape {idx} due to parsing errors.")
#             continue

#         print(f"✅ Extracted shape {idx} from the train dataset!")

#         # Create VTK PolyData
#         points = vtk.vtkPoints()
#         for point in world_pos:
#             points.InsertNextPoint(point.tolist())

#         # Create VTK cells (QUADS instead of triangles)
#         cells = vtk.vtkCellArray()
#         for cell in mesh_cells:
#             quad = vtk.vtkQuad()
#             for i in range(4):
#                 quad.GetPointIds().SetId(i, cell[i])
#             cells.InsertNextCell(quad)

#         # Create the VTK PolyData object
#         poly_data = vtk.vtkPolyData()
#         poly_data.SetPoints(points)
#         poly_data.SetPolys(cells)

#         # Convert and add node attributes to VTK
#         node_type_vtk = numpy_to_vtk(node_type, deep=True, array_type=vtk.VTK_INT)
#         node_type_vtk.SetName("node_type")
#         poly_data.GetPointData().AddArray(node_type_vtk)

#         stress_vtk = numpy_to_vtk(stress, deep=True, array_type=vtk.VTK_FLOAT)
#         stress_vtk.SetName("stress")
#         poly_data.GetPointData().AddArray(stress_vtk)

#         # Write to a .vtk file
#         output_vtk_file = os.path.join(OUTPUT_VTK_DIR, f"deforming_plate_shape_{idx}.vtk")
#         writer = vtk.vtkPolyDataWriter()
#         writer.SetFileName(output_vtk_file)
#         writer.SetInputData(poly_data)
#         writer.Write()

#         print(f"✅ Successfully saved shape {idx} as VTK file: {output_vtk_file}")

# except tf.errors.NotFoundError:
#     print(f"❌ ERROR: The file {TFRECORD_FILE} was not found.")
# except ValueError as ve:
#     print(f"❌ ERROR: {ve}")
# except Exception as e:
#     print(f"❌ ERROR: An unexpected error occurred: {e}")

# import tensorflow as tf
# import numpy as np
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk
# import os

# # Define paths for the dataset and output directory
# TFRECORD_FILE = r"C:\Users\Bazzi\Desktop\Mesh\deepmind-research\meshgraphnets\data\deforming_plate\train.tfrecord"
# OUTPUT_VTK_EVAL_DIR = r"C:\Users\Bazzi\Desktop\Mesh\deepmind-research\meshgraphnets\data\deforming_plate\time_train_outputs"

# # Ensure output directory exists
# os.makedirs(OUTPUT_VTK_EVAL_DIR, exist_ok=True)

# def parse_example(example_proto):
#     """Parses a single TFRecord example and extracts features."""
#     feature_description = {
#         'world_pos': tf.io.FixedLenFeature([], tf.string),
#         'mesh_pos': tf.io.FixedLenFeature([], tf.string),
#         'cells': tf.io.FixedLenFeature([], tf.string),
#         'node_type': tf.io.FixedLenFeature([], tf.string),
#         'stress': tf.io.FixedLenFeature([], tf.string),
#     }
#     parsed_features = tf.io.parse_single_example(example_proto, feature_description)

#     # Decode raw bytes
#     world_pos_raw = tf.io.decode_raw(parsed_features['world_pos'], tf.float32).numpy()
#     mesh_pos_raw = tf.io.decode_raw(parsed_features['mesh_pos'], tf.float32).numpy()
#     mesh_cells = tf.io.decode_raw(parsed_features['cells'], tf.int32).numpy()
#     node_type = tf.io.decode_raw(parsed_features['node_type'], tf.int32).numpy()
#     stress = tf.io.decode_raw(parsed_features['stress'], tf.float32).numpy()

#     try:
#         # Infer number of nodes (N) from mesh_pos (should be (N, 3))
#         if mesh_pos_raw.size % 3 != 0:
#             raise ValueError("mesh_pos size is not a multiple of 3")
#         N = mesh_pos_raw.size // 3

#         # Determine number of time steps (T) from world_pos
#         total = world_pos_raw.size
#         if total % (N * 3) != 0:
#             raise ValueError(f"world_pos size {total} is not a multiple of (N*3) where N is {N}")
#         T = total // (N * 3)

#         # Reshape arrays
#         world_pos = world_pos_raw.reshape(T, N, 3)
#         mesh_pos = mesh_pos_raw.reshape(N, 3)
#         mesh_cells = mesh_cells.reshape(-1, 4)
#         node_type = node_type.reshape(-1)
#         stress = stress.reshape(T, N)  # assuming stress varies over time

#     except Exception as e:
#         print(f"❌ ERROR in reshaping: {e}")
#         return None, None, None, None, None, None, None

#     return world_pos, mesh_pos, mesh_cells, node_type, stress, N, T

# # Range of shapes to extract
# start_idx = 2  # Start from shape index 2
# end_idx = 20   # Stop at shape index 20

# # Create TFRecordDataset from file
# dataset = tf.data.TFRecordDataset(TFRECORD_FILE)

# # Process dataset while keeping track of index
# for idx, raw_record in enumerate(dataset):
#     if idx < start_idx:
#         continue  # Skip shapes before index 2
#     if idx > end_idx:
#         break  # Stop once index exceeds 20

#     parsed = parse_example(raw_record)
#     if parsed[0] is None:
#         print(f"⚠️ Skipping shape {idx} due to parsing errors.")
#         continue

#     world_pos, mesh_pos, mesh_cells, node_type, stress, N, T = parsed
#     print(f"✅ Extracted rollout for shape {idx} with {T} time steps and {N} nodes!")

#     # Loop over each time step for this shape
#     for t in range(T):
#         current_world_pos = world_pos[t]
#         current_stress = stress[t]

#         # Create VTK PolyData for the current time step
#         points = vtk.vtkPoints()
#         for point in current_world_pos:
#             points.InsertNextPoint(point.tolist())

#         # Create VTK cells (assuming static connectivity)
#         cells = vtk.vtkCellArray()
#         for cell in mesh_cells:
#             quad = vtk.vtkQuad()
#             for i in range(4):
#                 quad.GetPointIds().SetId(i, cell[i])
#             cells.InsertNextCell(quad)

#         poly_data = vtk.vtkPolyData()
#         poly_data.SetPoints(points)
#         poly_data.SetPolys(cells)

#         # Add static attribute: node_type
#         node_type_vtk = numpy_to_vtk(node_type, deep=True, array_type=vtk.VTK_INT)
#         node_type_vtk.SetName("node_type")
#         poly_data.GetPointData().AddArray(node_type_vtk)

#         # Add time-varying attribute: stress
#         stress_vtk = numpy_to_vtk(current_stress, deep=True, array_type=vtk.VTK_FLOAT)
#         stress_vtk.SetName("stress")
#         poly_data.GetPointData().AddArray(stress_vtk)

#         # Optionally add time information as field data so ParaView recognizes it
#         time_value = np.array([t], dtype=np.float32)
#         time_vtk = numpy_to_vtk(time_value, deep=True, array_type=vtk.VTK_FLOAT)
#         time_vtk.SetName("TIME")
#         poly_data.GetFieldData().AddArray(time_vtk)

#         # Write each time step as a separate VTK file
#         output_vtk_file = os.path.join(OUTPUT_VTK_EVAL_DIR, f"deforming_plate_shape_{idx}_time_{t}.vtk")
#         writer = vtk.vtkPolyDataWriter()
#         writer.SetFileName(output_vtk_file)
#         writer.SetInputData(poly_data)
#         writer.Write()
#         print(f"✅ Saved shape {idx}, time {t} as {output_vtk_file}")


import meshio
import numpy as np
import sys
import os
import json
import tensorflow as tf

# Add repo to path if needed
sys.path.append(r"C:\Users\Bazzi\Desktop\Mesh Deforming\deepmind-research\meshgraphnets")  # Correct this path if needed
from dataset import load_dataset

# Load the dataset
dataset = load_dataset(r"C:\Users\Bazzi\Desktop\Mesh Deforming\data\deforming_plate", "train")

# Take a single example
example = next(iter(dataset))

# Optional: print keys and shapes
for k in example:
    print(f"{k}: shape {example[k].shape}")

# Select timestep
t_idx = 15

# Extract mesh data at timestep t_idx
points = example["mesh_pos"].numpy()[t_idx]             # (840, 3)
cells = example["cells"].numpy()                        # (2564, 4)
print("nikniknik:", cells.shape)
stress = example["stress"].numpy()[t_idx, :, 0]         # (840,)
node_type = example["node_type"].numpy()[t_idx, :, 0]   # (840,)

# Format for meshio
cells = [("tetra", cells)]  # "quad" is correct for deforming_plate
point_data = {"stress": stress, "node_type": node_type}

# Create and write mesh
mesh = meshio.Mesh(points, cells, point_data=point_data)
mesh.write("data.vtk")
print("✅ Exported data.vtk")
