import tensorflow as tf
import numpy as np
from common import NodeType

def _rollout(model, initial_state, num_steps, target_world_pos):
    """Roll out the model forward in time from the initial graph."""

    # Create NORMAL and OBSTACLE node masks
    node_type = initial_state['node_type'][:, :, 0]  # shape: [1, N]
    normal_mask = tf.equal(node_type, NodeType.NORMAL.value)
    normal_mask = tf.expand_dims(normal_mask, axis=-1)
    normal_mask = tf.repeat(normal_mask, repeats=3, axis=-1)  # shape: [1, N, 3]

    cur_pos = initial_state['world_pos']  # shape: [1, N, 3]

    trajectory = []
    cur_positions = []
    cur_velocities = []
    stress_trajectory = []

    for step in tf.range(num_steps):
        model_input = {
            **initial_state,
            'world_pos': cur_pos,
            'target|world_pos': target_world_pos[:, step]
        }

        outputs = model.model(model_input, training=False)

        pred_pos = outputs['world_pos']
        cur_position = outputs.get('cur_position', pred_pos)
        cur_velocity = outputs.get('cur_velocity', tf.zeros_like(pred_pos))
        stress = outputs.get('stress', tf.zeros_like(pred_pos))

        next_pos = tf.where(normal_mask, pred_pos, target_world_pos[:, step])

        trajectory.append(next_pos)
        cur_positions.append(cur_position)
        cur_velocities.append(cur_velocity)
        stress_trajectory.append(stress)

        cur_pos = tf.stop_gradient(next_pos)

    trajectory = tf.stack(trajectory, axis=1)          # [1, T, N, 3]
    cur_positions = tf.stack(cur_positions, axis=1)    # [1, T, N, 3]
    cur_velocities = tf.stack(cur_velocities, axis=1)  # [1, T, N, 3]
    stress_trajectory = tf.stack(stress_trajectory, axis=1)  # [1, T, N, 3]

    return trajectory, cur_positions, cur_velocities, stress_trajectory


def evaluate(model, batch):
    initial_state = batch
    target_world_pos = tf.expand_dims(batch['target|world_pos'], axis=1)  # [1, T=1, N, 3]
    num_steps = tf.shape(target_world_pos)[1]

    prediction, cur_positions, cur_velocities, stress = _rollout(
        model, initial_state, num_steps, target_world_pos
    )

    mse = tf.reduce_mean(tf.square(prediction - target_world_pos))
    l1 = tf.reduce_mean(tf.abs(prediction - target_world_pos))
    scalars = {
        'mse_world_pos': float(mse.numpy()),
        'l1_world_pos': float(l1.numpy())
    }

    faces = batch['cells'][0]  # remove batch dim
    face_start = faces[:, 0:3]
    face_end = tf.concat([faces[:, 2:], tf.expand_dims(faces[:, 0], axis=-1)], axis=-1)
    faces_result = tf.concat([face_start, face_end], axis=0)  # [2F, 3]

    traj_ops = {
        'faces': faces_result.numpy(),
        'mesh_pos': batch['mesh_pos'][0].numpy(),
        'gt_pos': batch['world_pos'][0].numpy(),
        'pred_pos': prediction[0].numpy(),
        'cur_positions': cur_positions[0].numpy(),
        'cur_velocities': cur_velocities[0].numpy(),
        'stress': stress[0].numpy()
    }

    return scalars, traj_ops
