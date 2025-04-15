import numpy as np
import tensorflow as tf
import os
import json

# Output directory
output_dir = "./dummy_deforming_plate_data"
os.makedirs(output_dir, exist_ok=True)

# Dummy size — much smaller than real data
trajectory_length = 10
num_nodes = 5
num_cells = 2

# Create dummy data
data = {
    "cells": np.random.randint(0, num_nodes, size=(1, num_cells, 4), dtype=np.int32),
    "node_type": np.random.randint(0, 2, size=(1, num_nodes, 1), dtype=np.int32),
    "mesh_pos": np.random.randn(1, num_nodes, 3).astype(np.float32),
    "world_pos": np.random.randn(trajectory_length, num_nodes, 3).astype(np.float32),
    "stress": np.random.randn(trajectory_length, num_nodes, 1).astype(np.float32)
}

# Save as TFRecord (VarLenFeature-compatible)
tfrecord_path = os.path.join(output_dir, "test.tfrecord")
with tf.io.TFRecordWriter(tfrecord_path) as writer:
    feature_dict = {}
    for key, array in data.items():
        serialized = tf.io.serialize_tensor(tf.convert_to_tensor(array)).numpy()
        feature_dict[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized])
        )
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    writer.write(example_proto.SerializeToString())

# Save meta.json (compatible with your parser)
meta = {
    "simulator": "comsol",
    "dt": 0,
    "collision_radius": 0.03,
    "features": {
        "cells": {
            "type": "static",
            "shape": [1, -1, 4],
            "dtype": "int32"
        },
        "node_type": {
            "type": "static",
            "shape": [1, -1, 1],
            "dtype": "int32"
        },
        "mesh_pos": {
            "type": "static",
            "shape": [1, -1, 3],
            "dtype": "float32"
        },
        "world_pos": {
            "type": "dynamic",
            "shape": [trajectory_length, -1, 3],
            "dtype": "float32"
        },
        "stress": {
            "type": "dynamic",
            "shape": [trajectory_length, -1, 1],
            "dtype": "float32"
        }
    },
    "field_names": ["cells", "node_type", "mesh_pos", "world_pos", "stress"],
    "trajectory_length": trajectory_length
}

meta_path = os.path.join(output_dir, "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Dummy TFRecord and meta.json created!")
print(f"TFRecord: {tfrecord_path}")
print(f"Meta:     {meta_path}")
