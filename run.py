import pickle
from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import importlib

import dataset
import core_model
import deform_eval  # evaluation module for deforming plate
import deform_model  # this module defines DeformingPlateModel

import csv
import os

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'deforming_plate'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoints')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None, 'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'Number of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'Number of training steps')

# Parameters for different models.
PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model_module='cfd_model', evaluator_module='cfd_eval'),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model_module='cloth_model', evaluator_module='cloth_eval'),
    'deforming_plate': dict(noise=0.003, gamma=0.2, field='world_pos', history=False,
                            size=4, batch=2, model_module='deform_model', evaluator_module='deform_eval')
}

import os
import csv
import numpy as np
import tensorflow as tf
import dataset


def learner(model, params):
    """Run the training loop with checkpointing and debugging prints."""

    log_path = "train_log.csv"

    # Create CSV header if file doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "velocity_loss", "stress_loss"])

    try:
        print("\n=== Loading Dataset ===")
        print(f"Loading from: {FLAGS.dataset_dir}")
        ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')

        print("\n=== Sample Structure ===")
        for sample in ds.take(1):
            print("\nFirst sample details:")
            for k, v in sample.items():
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")

        print("\n=== Adding Targets ===")
        ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
        print("Targets added successfully")

        print("\n=== Preprocessing Data ===")
        ds = dataset.split_and_preprocess(
            ds,
            noise_field=params['field'],
            noise_scale=params['noise'],
            noise_gamma=params['gamma']
        )
        print("Preprocessing completed")

        print("\n=== Setting Up Training ===")
        iterator = iter(ds)
        global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        print(f"Initial global step: {global_step.numpy()}")

        def compute_loss(inputs):
            return model.loss(inputs)

        initial_learning_rate = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=int(5e6),
            decay_rate=0.1,
            staircase=True
        )

        print("\n=== Creating Optimizer ===")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("Optimizer created successfully")

        # === Checkpointing ===
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, max_to_keep=5)

        # Restore latest checkpoint if available
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f"✅ Restored checkpoint from {ckpt_manager.latest_checkpoint}")
        else:
            print("ℹ️ No checkpoint found — training from scratch.")

        @tf.function(reduce_retracing=True)
        def train_step(inputs):
            with tf.GradientTape() as tape:
                loss = compute_loss(inputs)

            def skip_step():
                tf.print("⚠️ Skipping step: Loss is NaN or Inf —", loss)
                return tf.constant(float('nan'), dtype=tf.float32)

            def apply_step():
                grads = tape.gradient(loss, model.trainable_variables)
                clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
                optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
                return loss

            is_bad_loss = tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss))
            return tf.cond(is_bad_loss, skip_step, apply_step)

        loss_history = []
        steps_per_epoch = 10
        start_time = tf.timestamp()
        current_loss = None
        warmup_steps = 10

        print("\n=== Starting Training Loop ===")
        for step in range(global_step.numpy(), FLAGS.num_training_steps):
            try:
                print(f"\nStep {step + 1}/{FLAGS.num_training_steps}")
                inputs = next(iterator)

                if global_step < warmup_steps:
                    global_step.assign_add(1)
                    print(f"Warmup step {global_step.numpy()}/{warmup_steps}")
                    continue

                loss = train_step(inputs)
                if loss is not None:
                    current_loss = tf.get_static_value(loss)
                    stress_val = tf.get_static_value(model.last_stress_loss) if hasattr(model, "last_stress_loss") else 0.0

                    loss_history.append(current_loss)
                    step_num = global_step.numpy()
                    global_step.assign_add(1)

                    # Write to CSV for Streamlit
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([step_num, current_loss, stress_val])

                    if tf.equal(global_step % steps_per_epoch, 0):
                        current_time = tf.timestamp()
                        elapsed_time = current_time - start_time
                        steps_per_sec = steps_per_epoch / elapsed_time
                        avg_loss = np.mean(loss_history[-steps_per_epoch:])

                        print(f"\n=== Training Progress at Step {step_num} ===")
                        print(f"Current Loss: {current_loss:.6f}")
                        print(f"Average Loss (last {steps_per_epoch} steps): {avg_loss:.6f}")
                        current_lr = lr_schedule(global_step.numpy())
                        print(f"Learning Rate: {float(current_lr.numpy()):.6f}")
                        print(f"Steps per second: {steps_per_sec:.2f}")
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")

                        start_time = tf.timestamp()

                        try:
                            import psutil
                            process = psutil.Process()
                            memory_info = process.memory_info()
                            print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                        except ImportError:
                            pass

                    # Save checkpoint every 1000 steps
                    if step_num % 1000 == 0:
                        ckpt_path = ckpt_manager.save()
                        print(f"✅ Saved checkpoint at step {step_num}: {ckpt_path}")
                else:
                    print("Skipping step due to invalid loss")
            except Exception as e:
                print(f"Error in training step {step}: {str(e)}")
                break

        print("\n=== Training Complete ===")
        if current_loss is not None:
            print(f"Final Loss: {current_loss:.6f}")
        print(f"Total Steps: {global_step.numpy()}")
        if loss_history:
            print(f"Average Loss (all steps): {np.mean(loss_history):.6f}")
        final_lr = lr_schedule(global_step.numpy())
        print(f"Final Learning Rate: {float(final_lr.numpy()):.6f}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise



def evaluator(model, params):
    """Run model rollout and compute evaluation metrics in TF2."""

    # === Load dataset ===
    ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])

    # Pad dataset to batch size 1
    padded_shapes = tf.nest.map_structure(lambda spec: spec.shape, ds.element_spec)
    ds = ds.padded_batch(1, padded_shapes=padded_shapes)

    # === Restore checkpoint ===
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        logging.info("Restored from checkpoint: %s", ckpt_manager.latest_checkpoint)
    else:
        logging.info("No checkpoint found. Evaluating with current model weights.")

    # === Load evaluator module ===
    import importlib
    evaluator_module = importlib.import_module(params['evaluator_module'])

    trajectories = []
    mse_losses = []
    l1_losses = []

    for i, batch in enumerate(ds.take(FLAGS.num_rollouts)):
        logging.info("Evaluating rollout %d", i)

        # Run evaluation on one batch
        scalar_data, traj_ops = evaluator_module.evaluate(model, batch)

        # Log scalar metrics
        mse_losses.append(scalar_data['mse_world_pos'])
        l1_losses.append(scalar_data['l1_world_pos'])
        logging.info("    MSE loss: %f", scalar_data['mse_world_pos'])
        logging.info("    L1  loss: %f", scalar_data['l1_world_pos'])

        trajectories.append(traj_ops)

    # === Report aggregate losses ===
    avg_mse = np.mean(mse_losses)
    avg_l1 = np.mean(l1_losses)
    logging.info("==== Evaluation Summary ====")
    logging.info("Mean MSE: %f", avg_mse)
    logging.info("Mean L1 : %f", avg_l1)

    # === Save trajectory outputs ===
    with open(FLAGS.rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
    logging.info("Saved rollout trajectories to %s", FLAGS.rollout_path)

def main(argv):
    del argv
    print("\n=== Initializing Model ===")
    params = PARAMETERS[FLAGS.model]
    print(f"Model parameters: {params}")
    
    print("\n=== Creating Core Model ===")
    # Instantiate the learned model from core_model.
    learned_model = core_model.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=10)
    print("Core model created successfully")
    
    print("\n=== Creating Deforming Plate Model ===")
    if FLAGS.model == 'deforming_plate':
        from deform_model import DeformingPlateModel
        model = DeformingPlateModel(learned_model)
        print("Deforming plate model created successfully")
    else:
        raise ValueError('Invalid model type for this run; only deforming_plate is supported.')
    
    print("\n=== Starting Training Process ===")
    if FLAGS.mode == 'train':
        print(f"Training for {FLAGS.num_training_steps} steps")
        print(f"Dataset directory: {FLAGS.dataset_dir}")
        print(f"Checkpoint directory: {FLAGS.checkpoint_dir}")
        learner(model, params)
    elif FLAGS.mode == 'eval':
        print("Starting evaluation")
        evaluator(model, params)

if __name__ == '__main__':
    app.run(main)
