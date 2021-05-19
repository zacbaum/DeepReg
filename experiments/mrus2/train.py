import os
import math
import tensorflow as tf
import random
from datetime import datetime
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import deepreg.model.optimizer as opt
import deepreg.model.layer_util as layer_util
from deepreg.model.layer import Warping
from deepreg.registry import REGISTRY, Registry
from deepreg.train import build_config
from deepreg.util import build_dataset, build_log_dir, calculate_metrics
from deepreg.callback import build_checkpoint_callback

@tf.function
def train_step(train_input, fixed_lb):

    with tf.GradientTape() as tape:

        pred = model(train_input, training=True)
        pred_lb = pred["pred_fixed_label"]

        label_loss = multi_scale_loss(fixed_lb, pred_lb, [0, 1, 2, 4, 8, 16, 32], dice_simple)
        if tf.math.is_nan(label_loss):
            label_loss = tf.constant(0, dtype=tf.float32)
        
        regularizer_loss = tf.reduce_mean(model._model.losses[1]) # Get built-in deepreg bending energy.

        loss = gamma * label_loss + alpha * regularizer_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, label_loss, regularizer_loss

def sample_input_slices(arr, slices):

    masked_arr = np.zeros(arr.shape)
    for s in slices: 
        masked_arr[:, s, :] = arr[:, s, :]
    return masked_arr

def mask_inputs(train_input, slices):

    if slices is not None:

        fixed_im = train_input["fixed_image"]
        masked_fixed_im = np.empty(fixed_im.shape)
        for i in range(len(slices)):
            masked_fixed_im[i] = sample_input_slices(fixed_im[i], slices[i])
        
        fixed_lb = train_input["fixed_label"]
        masked_fixed_lb = np.empty(fixed_lb.shape)
        for i in range(len(slices)):
            if train_input["indices"][i][1] == 0:
                masked_fixed_lb[i] = sample_input_slices(fixed_lb[i], slices[i])
        
        masked_fixed_im = tf.convert_to_tensor(masked_fixed_im, dtype=tf.float32)
        masked_fixed_lb = tf.convert_to_tensor(masked_fixed_lb, dtype=tf.float32)
        x = {"fixed_image": masked_fixed_im, "fixed_label": masked_fixed_lb}
        train_input.update(x)

    return train_input

def gauss_kernel1d(sigma):

    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = tf.exp([-0.5*x**2/sigma**2 for x in range(-tail, tail+1)])
        return k / tf.reduce_sum(k)

def separable_filter3d(vol, kernel):
    
    if np.isscalar(kernel) and kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,
            tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME")

def multi_scale_loss(label_fixed, label_moving, loss_scales, loss_type):
    
    label_fixed = tf.expand_dims(label_fixed, axis=-1)
    label_moving = tf.expand_dims(label_moving, axis=-1)
    losses = []
    for s in loss_scales:
        if s == 0:
            losses.append(
                loss_type(
                    label_fixed,
                    label_moving
                )
            )
        else:
            losses.append(
                loss_type(
                    separable_filter3d(label_fixed, gauss_kernel1d(s)),
                    separable_filter3d(label_moving, gauss_kernel1d(s))
                )
            )
    return tf.add_n(losses) / len(losses)

def dice_simple(y_true, y_pred, eps=tf.keras.backend.epsilon()):
    
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    dsc = -((numerator + eps) / (denominator + eps))
    return  tf.reduce_mean(dsc)

def evaluate_val_data(dataset_train, dataset_metrics, iteration, frames, plot=False, use_meta=True):

    dice_scores = [[] for _ in range(frames - 1)] if (frames > 0) else [[]]
    tres = [[] for _ in range(frames - 1)] if (frames > 0) else [[]]

    for _, train_inputs in enumerate(dataset_train):

        if train_inputs['indices'].numpy()[0][1] == 0:

            # Get the base fixed images
            full_fixed_im = train_inputs["fixed_image"]
            full_fixed_lb = train_inputs["fixed_label"]

            sh = tf.shape(full_fixed_im).numpy()[1:]
            fgr = tf.expand_dims(layer_util.get_reference_grid(sh), axis=0)

            # Get the 2-slice input for the zero-shot metrics and train-step
            if frames > 0:
                n_slices = 2
                start = tf.shape(train_inputs["fixed_image"])[2] // 6
                stop = 5 * tf.shape(train_inputs["fixed_image"])[2] // 6
                slices = [np.random.choice(list(range(start, stop)), size=(n_slices), replace=False)]
            else:
                slices = None
            eval_inputs = mask_inputs(deepcopy(train_inputs), slices)
            eval_pred = model(eval_inputs, training=False)
            
            # Compute all zero-shot metrics
            for _, metrics_inputs in enumerate(dataset_metrics):

                if eval_inputs['indices'].numpy()[0][0] == metrics_inputs['indices'].numpy()[0][0]:

                    t_label = Warping(fixed_image_size=sh)([eval_pred["ddf"], metrics_inputs["moving_label"]])
                    metrics = calculate_metrics(
                        fixed_image=metrics_inputs["fixed_image"],
                        fixed_label=metrics_inputs["fixed_label"],
                        pred_fixed_image=None,
                        pred_fixed_label=t_label,
                        fixed_grid_ref=fgr,
                        sample_index=0,
                    )  

                    if metrics_inputs['indices'].numpy()[0][1] == 0:
                        dice_scores[0].append(metrics["label_binary_dice"])
                    tres[0].append(metrics["label_tre"])

                    if metrics_inputs["indices"].numpy()[0][0] <= 3 and plot:
                        plot_helper(
                            metrics_inputs["indices"].numpy(),
                            source=metrics_inputs["moving_image"].numpy(),
                            source_label=metrics_inputs["moving_label"].numpy(),
                            transformed=eval_pred["pred_fixed_image"].numpy(),
                            transformed_label=t_label.numpy(),
                            target=eval_inputs["fixed_image"].numpy(),
                            target_label=eval_inputs["fixed_label"].numpy() if metrics_inputs["indices"].numpy()[0][1] == 0 else metrics_inputs["fixed_label"].numpy(),
                            target_full=full_fixed_im.numpy(),
                            target_label_full=full_fixed_lb.numpy(),
                            iteration=iteration,
                            shot=0,
                        )

            if frames > 0:

                weights_before = model.get_weights()

                for frame in range(3, frames + 1):

                    if use_meta:
                        # Do (n-1)-shot training step before getting new metrics
                        train_step(eval_inputs, full_fixed_lb)

                    # Get the n-slice input (1 new slice)
                    while len(slices[0]) < frame:
                        s = np.random.choice(list(range(start, stop)), size=(1))
                        if not s in slices[0]:
                            slices[0] = np.append(slices, s)
                    
                    eval_inputs = mask_inputs(deepcopy(train_inputs), slices)
                    eval_pred = model(eval_inputs, training=False)
                    
                    # Compute all n-shot metrics
                    for _, metrics_inputs in enumerate(dataset_metrics):
                        
                        if eval_inputs['indices'].numpy()[0][0] == metrics_inputs['indices'].numpy()[0][0]:
                        
                            t_label = Warping(fixed_image_size=sh)([eval_pred["ddf"], metrics_inputs["moving_label"]])
                            metrics = calculate_metrics(
                                fixed_image=metrics_inputs["fixed_image"],
                                fixed_label=metrics_inputs["fixed_label"],
                                pred_fixed_image=None,
                                pred_fixed_label=t_label,
                                fixed_grid_ref=fgr,
                                sample_index=0,
                            )   
                            # Append to (frame-2) to align 2 frames as 0th index (3 frames as 1st, ..., 10 frames as 8th)
                            if metrics_inputs['indices'].numpy()[0][1] == 0:
                                dice_scores[frame - 2].append(metrics["label_binary_dice"])
                            tres[frame - 2].append(metrics["label_tre"])

                            if metrics_inputs["indices"].numpy()[0][0] <= 3 and plot:
                                plot_helper(
                                    metrics_inputs["indices"].numpy(),
                                    source=metrics_inputs["moving_image"].numpy(),
                                    source_label=metrics_inputs["moving_label"].numpy(),
                                    transformed=eval_pred["pred_fixed_image"].numpy(),
                                    transformed_label=t_label,
                                    target=eval_inputs["fixed_image"].numpy(),
                                    target_label=eval_inputs["fixed_label"].numpy() if metrics_inputs["indices"].numpy()[0][1] == 0 else metrics_inputs["fixed_label"].numpy(),
                                    target_full=full_fixed_im.numpy(),
                                    target_label_full=full_fixed_lb.numpy(),
                                    iteration=iteration,
                                    shot=(frame - 2),
                                )

                if use_meta:
                    # Reset weights for next one-shot learning step
                    model.set_weights(weights_before)

    return dice_scores, tres

def plot_helper(indices, source, source_label, transformed, transformed_label, target, target_label, target_full, target_label_full, iteration, shot):

    nrow = 15
    ncol = 4
    offset = 10

    fig = plt.figure(figsize=(ncol+1, nrow+1)) 
    gs = matplotlib.gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.05, top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

    indxs = np.linspace(offset, np.squeeze(target.shape[1]) - (1 + offset), nrow)
    indxs_cnt = 0

    for i in range(0, nrow*ncol, ncol):
        ax1 = plt.subplot(gs[i])
        ax2 = plt.subplot(gs[i+1])
        ax3 = plt.subplot(gs[i+2])
        ax4 = plt.subplot(gs[i+3])
        
        ax1.grid(False)
        ax1.set_axis_off()
        ax2.grid(False)
        ax2.set_axis_off()
        ax3.grid(False)
        ax3.set_axis_off()
        ax4.grid(False)
        ax4.set_axis_off()

        ax1.imshow(np.rot90(np.squeeze(source)[int(indxs[indxs_cnt])], 3), cmap='gray')
        ax1.imshow(np.rot90(np.squeeze(source_label)[int(indxs[indxs_cnt])], 3), alpha=0.2, cmap='Reds', vmin=0, vmax=2)

        ax2.imshow(np.rot90(np.squeeze(transformed)[int(indxs[indxs_cnt])], 3), cmap='gray')
        ax2.imshow(np.rot90(np.squeeze(transformed_label)[int(indxs[indxs_cnt])], 3), alpha=0.2, cmap='Reds', vmin=0, vmax=2)

        ax3.imshow(np.rot90(np.squeeze(target_full)[int(indxs[indxs_cnt])], 3), cmap='gray')  
        if indices[0][1] == 0:
            ax3.imshow(np.rot90(np.squeeze(target_label_full)[int(indxs[indxs_cnt])], 3), alpha=0.2, cmap='Reds', vmin=0, vmax=2)
        else:
            ax3.imshow(np.rot90(np.squeeze(target_label)[int(indxs[indxs_cnt])], 3), alpha=0.2, cmap='Reds', vmin=0, vmax=2)

        ax4.imshow(np.rot90(np.squeeze(target)[int(indxs[indxs_cnt])], 3), cmap='gray')    
        ax4.imshow(np.rot90(np.squeeze(target_label)[int(indxs[indxs_cnt])], 3), alpha=0.2, cmap='Reds', vmin=0, vmax=2)

        indxs_cnt += 1

    plt.savefig(os.path.join(log_dir, "iter-{:06}_{}-shot-output_sample-{}-lm-{}.png".format(iteration, shot, int(indices[0][0]), int(indices[0][1]))))
    plt.close()

gpu="3"
gpu_allow_growth=True
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if gpu_allow_growth else "false"
matplotlib.rcParams['agg.path.chunksize'] = 10000

config_path="fold00.yaml"
ckpt_path=""#"/raid/candi/zbaum/logs/.../save/ckpt-2100"
log_dir="/raid/candi/zbaum/logs"
exp_name="3d-2d-a1g1-10shot"
registry=REGISTRY

# load config
config, log_dir, ckpt_path = build_config(
    config_path=config_path,
    log_dir=log_dir,
    exp_name=exp_name,
    ckpt_path=ckpt_path,
)
# Create validation and test configs where we use batch size of 1
eval_config = deepcopy(config)
eval_config["train"]["preprocess"]["batch_size"] = 1

# build dataset
data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
    dataset_config=config["dataset"],
    preprocess_config=config["train"]["preprocess"],
    mode="train",
    training=True,
    repeat=True,
)

data_loader_val, dataset_val, steps_per_epoch_val = build_dataset(
    dataset_config=eval_config["dataset"],
    preprocess_config=eval_config["train"]["preprocess"],
    mode="valid",
    training=False,
    repeat=False,
)

data_loader_test, dataset_test, _ = build_dataset(
    dataset_config=eval_config["dataset"],
    preprocess_config=eval_config["train"]["preprocess"],
    mode="test",
    training=False,
    repeat=False,
)

batch_size=config["train"]["preprocess"]["batch_size"]
strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = registry.build_model(
        config=dict(
            name=config["train"]["method"],
            moving_image_size=data_loader_train.moving_image_shape,
            fixed_image_size=data_loader_train.fixed_image_shape,
            index_size=data_loader_train.num_indices,
            labeled=config["dataset"]["labeled"],
            batch_size=batch_size,
            config=config["train"],
        )
    )
    optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])

# compile
model.compile(optimizer=optimizer)

# load weights
start_step = 0
if ckpt_path != "":

    model.fit(x=dataset_train, steps_per_epoch=1, epochs=1, verbose=0)
    model.load_weights(ckpt_path)

    # For deepreg models
    '''
    _, _ = build_checkpoint_callback(
        model=model,
        dataset=dataset_train,
        log_dir=log_dir,
        save_period=config["train"]["save_period"],
        ckpt_path=ckpt_path,
    )
    '''
    start_step = 0

#########################
#                       #
#    TRAINING LOOPS     #
#                       #
#########################
total_steps = config["train"]["epochs"]
save_period = config["train"]["save_period"]
inner_steps_per_meta_step = config["train"]["k"]
meta_step_lr = config["train"]["meta_lr"]
frames = config["train"]["frames"]
use_meta = config["train"]["use_meta"]
gamma = config["train"]["loss"]["label"]["weight"] # label loss weight
alpha = config["train"]["loss"]["regularization"]["weight"] # regularizer loss weight

print("###################################")
print("LOG: ", log_dir)
print("GPU: ", gpu)
print("ITERATIONS:", total_steps)
print("MAX FRAMES:", frames)
if use_meta:
    print("META-STEP SIZE: ", inner_steps_per_meta_step)
    print("INITIAL META LR: ", meta_step_lr)
print("OPT: ", config["train"]["optimizer"])
print("LABEL LOSS WEIGHT:", gamma)
print("DDF LOSS WEIGHT:", alpha)
print("###################################")
print()

evaluation = False
if evaluation:
    dice_scores, tres = evaluate_val_data(dataset_val, dataset_test, 0, True)
    '''
    print(
        "Evaluation:\nZero-Shot -- Dice: {:.3f} | Avg. TRE: {:.3f} | Med. TRE: {:.3f}\nOne-Shot --- Dice: {:.3f} | Avg. TRE: {:.3f} | Med. TRE: {:.3f}\n".format(
                                                                                                                np.mean(dice_scores_zero_shot), 
                                                                                                                np.mean(tres_zero_shot), 
                                                                                                                np.median(tres_zero_shot), 
                                                                                                                np.mean(dice_scores_one_shot), 
                                                                                                                np.mean(tres_one_shot),
                                                                                                                np.median(tres_one_shot),
                                                                                                            )
    )  
    '''    

iters = []
dice_loss = []
dice_loss_temp = []
be_loss = []
be_loss_temp = []

evals = []
mean_dice = [[] for _ in range(frames - 1)] if (frames > 0) else [[]]
mean_tres = [[] for _ in range(frames - 1)] if (frames > 0) else [[]]
median_tres = [[] for _ in range(frames - 1)] if (frames > 0) else [[]]

# Temporarily save the weights from the model.
weights_before = model.get_weights()

train_iterator = iter(dataset_train)

for iteration in range(start_step, total_steps):
    
    train_input = next(train_iterator)

    # Get the full label before removing data.
    fixed_lb = train_input["fixed_label"]

    # Remove the data from all but the selected slices, preserve full mask if not gland labels.
    # TODO: Should this be randomly selected for each set of training steps? Or always be random... i.e. one Reptile update PER task instead of mixing altogether...?
    if frames > 0:
        slices = []
        for b in range(batch_size):
            n_slices = np.random.choice(list(range(2, frames + 1)))
            start = tf.shape(train_input["fixed_image"])[2] // 6
            stop = 5 * tf.shape(train_input["fixed_image"])[2] // 6
            slices.append(np.random.choice(list(range(start, stop)), size=(n_slices), replace=False))
    else:   
        slices = None
    
    train_input = mask_inputs(deepcopy(train_input), slices)
    loss, label_loss, regularizer_loss = train_step(train_input, fixed_lb)

    for m in model._model.metrics:
        if m.name in "metric/TRE":
            train_tre = m.result()
    model._model.reset_metrics()

    dice_loss_temp.append(label_loss)
    be_loss_temp.append(regularizer_loss)

    print("Iteration: {:6d} | Loss: {:.5f} - Dice Loss: {:.5f} - BE Loss: {:.5f} - TRE: {:.3f} ".format(iteration+1, loss.numpy(), label_loss.numpy(), regularizer_loss.numpy(), train_tre.numpy()))

    if iteration >= 0 and (iteration + 1) % inner_steps_per_meta_step == 0:

        if use_meta:
            # Linear scheduler for reptile meta-SGD step.
            frac_done = iteration / total_steps
            cur_meta_step_lr = (1 - frac_done) * meta_step_lr
            print("Iteration: Meta-Update | Meta LR: {:.5f}".format(cur_meta_step_lr))
            
            # Get the non-interactively trained weights so far.
            weights_after = model.get_weights()

            # Perform SGD for the meta-step, load the newly-trained weights into the model.
            model.set_weights([weights_before[i] + (weights_after[i] - weights_before[i]) * cur_meta_step_lr for i in range(len(model.weights))])

            # Temporarily save the weights from the model (so we can restore after meta-testing).
            weights_before = model.get_weights()

        if (iteration + 1) % save_period == 0:

            dice_scores, tres = evaluate_val_data(dataset_val, dataset_test, iteration + 1, frames, plot=True, use_meta=use_meta)
            assert len(dice_scores) == len(tres)
            print("\nEvaluation:")
            for shot in range(len(dice_scores)):
                mean_dice[shot].append(np.mean(dice_scores[shot]))
                mean_tres[shot].append(np.mean(tres[shot]))
                median_tres[shot].append(np.median(tres[shot]))
                print("{}-Shot -- Dice: {:.3f} | Avg. TRE: {:.3f} | Med. TRE: {:.3f}".format(
                                                                                               shot,
                                                                                               mean_dice[shot][-1], 
                                                                                               mean_tres[shot][-1], 
                                                                                               median_tres[shot][-1], 
                                                                                              )
                )
            print()
            
            if use_meta:
                # Reset weights, continue non-interaction training.
                model.set_weights(weights_before)

            # Record training iteration losses.
            iters.append(iteration + 1)
            dice_loss.append(np.mean(dice_loss_temp))
            be_loss.append(np.mean(be_loss_temp))
            dice_loss_temp = []
            be_loss_temp = []

            f, ax = plt.subplots(1, 1)
            ax.plot(iters, dice_loss, label="Dice Loss")        
            ax.plot(iters, be_loss, label="Bending Energy")        
            ax.legend(loc='upper right')     
            plt.show()
            plt.savefig(os.path.join(log_dir, 'live_loss.png'), dpi=1000)
            plt.close()

            f, ax = plt.subplots(1, 1)
            for shot in range(len(dice_scores)):
                ax.plot(iters, mean_dice[shot], label="{}-Shot Dice".format(shot))        
            ax.legend(loc='lower right')     
            plt.show()
            plt.savefig(os.path.join(log_dir, 'live_dice.png'), dpi=1000)
            plt.close()

            f, ax = plt.subplots(1, 1)
            for shot in range(len(dice_scores)):
                ax.plot(iters, mean_tres[shot], label="Mean {}-Shot TRE".format(shot)) 
                ax.plot(iters, median_tres[shot], label="Median {}-Shot TRE".format(shot)) 
            ax.legend(loc='upper right')     
            plt.show()
            plt.savefig(os.path.join(log_dir, 'live_tre.png'), dpi=1000)
            plt.close()

            model.save_weights(os.path.join(log_dir, 'model'))

# close file loaders in data loaders after training
data_loader_train.close()
if data_loader_val is not None:
    data_loader_val.close()
if data_loader_test is not None:
    data_loader_test.close()
