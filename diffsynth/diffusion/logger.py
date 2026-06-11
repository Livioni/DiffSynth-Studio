import os, torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.video_warning_printed = False
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def log_video(self, key, path, step, fps=4):
        try:
            import imageio.v2 as imageio
            import numpy as np

            reader = imageio.get_reader(path)
            try:
                frames = [frame for frame in reader]
            finally:
                reader.close()
            if len(frames) == 0:
                return
            video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0)
            video = video.to(dtype=torch.float32) / 255.0
            self.writer.add_video(key, video, step, fps=fps)
        except Exception as error:
            if not self.video_warning_printed:
                print(f"Warning: TensorBoard video logging failed, skip videos. Error: {error}")
                self.video_warning_printed = True

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import swanlab
        project_name = os.environ.get("SWANLAB_PROJECT", project_name)
        self.swanlab = swanlab
        self.swanlab.init(project=project_name, logdir=log_dir)
        print(f"SwanLab is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.swanlab.log({key: value}, step=step)

    def close(self):
        self.swanlab.finish()


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None, run_name=None):
        import wandb
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, dir=log_dir, name=run_name)
        print(f"Wandb is enabled. Project: {project_name}; Run: {run_name}")

    def log(self, key, value, step):
        self.wandb.log({key: value}, step=step)

    def log_video(self, key, path, step, fps=4):
        # The frame rate is encoded when the video file is saved. Passing fps
        # alongside a file path is ignored by wandb and emits a warning.
        self.wandb.log({key: self.wandb.Video(path, format="mp4")}, step=step)

    def close(self):
        self.wandb.finish()


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=True,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        self.loggers = []
        self.loggers_initialized = False

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            wandb_run_name = os.path.basename(os.path.normpath(self.output_path))
            self.loggers.append(WandbLogger(project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log"), run_name=wandb_run_name))
        self.loggers_initialized = True

    def ensure_loggers_initialized(self):
        if not self.loggers_initialized:
            self.init_loggers()

    def log_metrics(self, metrics, step=None):
        self.ensure_loggers_initialized()
        step = self.num_steps if step is None else step
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            for logger in self.loggers:
                logger.log(key, value, step)

    def log_videos(self, videos, step=None, fps=4):
        self.ensure_loggers_initialized()
        step = self.num_steps if step is None else step
        for key, path in videos.items():
            for logger in self.loggers:
                if hasattr(logger, "log_video"):
                    logger.log_video(key, path, step, fps=fps)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, log_steps=1, **kwargs):
        self.num_steps += 1
        if accelerator.is_main_process:
            self.ensure_loggers_initialized()
            should_log = log_steps is None or log_steps <= 0 or self.num_steps % log_steps == 0
            if should_log:
                metrics = {}
                loss = kwargs.get("loss")
                if loss is not None:
                    metrics["train/loss"] = loss
                learning_rate = kwargs.get("learning_rate")
                if learning_rate is not None:
                    metrics["train/learning_rate"] = learning_rate
                if metrics:
                    self.log_metrics(metrics, step=self.num_steps)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        for logger in self.loggers:
            logger.close()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
