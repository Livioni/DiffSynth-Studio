import json, os, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
try:
    from accelerate import skip_first_batches
except ImportError:
    skip_first_batches = None
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


TRAINING_STATE_FILE = "training_state.json"
LATEST_TRAINING_CHECKPOINT_FILE = "latest"


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def get_training_checkpoint_root(output_path, training_checkpoint_dir=None):
    if training_checkpoint_dir is not None and training_checkpoint_dir != "":
        return training_checkpoint_dir
    return os.path.join(output_path, "training_checkpoints")


def resolve_training_checkpoint_path(path, output_path, training_checkpoint_dir=None):
    if path is None or path == "":
        return None
    if path in ("latest", "last"):
        root = get_training_checkpoint_root(output_path, training_checkpoint_dir)
        latest_path = os.path.join(root, LATEST_TRAINING_CHECKPOINT_FILE)
        if not os.path.isfile(latest_path):
            raise FileNotFoundError(f"No latest training checkpoint marker found at {latest_path}.")
        with open(latest_path, "r") as f:
            path = f.read().strip()
    return path


def save_training_checkpoint(
    accelerator: Accelerator,
    output_path: str,
    training_checkpoint_dir: str,
    global_step: int,
    epoch_id: int,
    batch_id: int,
    batches_per_epoch: int,
    num_epochs: int,
):
    checkpoint_root = get_training_checkpoint_root(output_path, training_checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_root, f"step-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(checkpoint_dir)
    next_batch_in_epoch = batch_id + 1
    next_epoch = epoch_id
    if batches_per_epoch is not None and next_batch_in_epoch >= batches_per_epoch:
        next_epoch += 1
        next_batch_in_epoch = 0
    metadata = {
        "global_step": int(global_step),
        "epoch": int(epoch_id),
        "batch_in_epoch": int(batch_id),
        "next_epoch": int(next_epoch),
        "next_batch_in_epoch": int(next_batch_in_epoch),
        "batches_per_epoch": None if batches_per_epoch is None else int(batches_per_epoch),
        "num_epochs": int(num_epochs),
    }
    if accelerator.is_main_process:
        with open(os.path.join(checkpoint_dir, TRAINING_STATE_FILE), "w") as f:
            json.dump(metadata, f, indent=2)
        os.makedirs(checkpoint_root, exist_ok=True)
        with open(os.path.join(checkpoint_root, LATEST_TRAINING_CHECKPOINT_FILE), "w") as f:
            f.write(os.path.abspath(checkpoint_dir))
    accelerator.wait_for_everyone()


def load_training_checkpoint(
    accelerator: Accelerator,
    output_path: str,
    training_checkpoint_dir: str,
    resume_training_checkpoint: str,
):
    checkpoint_dir = resolve_training_checkpoint_path(resume_training_checkpoint, output_path, training_checkpoint_dir)
    metadata_path = os.path.join(checkpoint_dir, TRAINING_STATE_FILE)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Training checkpoint directory does not exist: {checkpoint_dir}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Training checkpoint metadata does not exist: {metadata_path}")
    accelerator.load_state(checkpoint_dir)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if accelerator.is_main_process:
        print(f"Loaded training checkpoint from {checkpoint_dir} at step {metadata['global_step']}.")
    return metadata


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    log_steps: int = 1,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    eval_callback = None,
    save_training_checkpoint_enabled: bool = False,
    resume_training_checkpoint: str = None,
    training_checkpoint_dir: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        log_steps = getattr(args, "log_steps", log_steps)
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
        save_training_checkpoint_enabled = getattr(args, "save_training_checkpoint", save_training_checkpoint_enabled)
        resume_training_checkpoint = getattr(args, "resume_training_checkpoint", resume_training_checkpoint)
        training_checkpoint_dir = getattr(args, "training_checkpoint_dir", training_checkpoint_dir)

    should_save_training_checkpoint = save_training_checkpoint_enabled and save_steps is not None and save_steps > 0
    if (should_save_training_checkpoint or resume_training_checkpoint is not None) and enable_model_cpu_offload:
        raise ValueError("Full training checkpoints are not supported with --enable_model_cpu_offload.")

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    try:
        batches_per_epoch = len(dataloader)
    except TypeError:
        batches_per_epoch = None

    initialize_deepspeed_gradient_checkpointing(accelerator)
    start_epoch = 0
    start_batch_in_epoch = 0
    if resume_training_checkpoint is not None:
        metadata = load_training_checkpoint(
            accelerator,
            model_logger.output_path,
            training_checkpoint_dir,
            resume_training_checkpoint,
        )
        model_logger.num_steps = int(metadata["global_step"])
        start_epoch = int(metadata.get("next_epoch", 0))
        start_batch_in_epoch = int(metadata.get("next_batch_in_epoch", 0))

    for epoch_id in range(start_epoch, num_epochs):
        skip_batches = start_batch_in_epoch if epoch_id == start_epoch else 0
        if skip_batches > 0 and accelerator.is_main_process:
            print(f"Skip {skip_batches} already-trained batches in epoch {epoch_id}.")
        epoch_dataloader = dataloader
        batch_id_offset = 0
        if skip_batches > 0 and skip_first_batches is not None:
            epoch_dataloader = skip_first_batches(dataloader, skip_batches)
            batch_id_offset = skip_batches
        for batch_id, data in enumerate(tqdm(epoch_dataloader)):
            batch_id += batch_id_offset
            if skip_first_batches is None and batch_id < skip_batches:
                continue
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                learning_rate = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                model_logger.on_step_end(accelerator, model, save_steps, log_steps=log_steps, loss=loss, learning_rate=learning_rate)
                if (
                    should_save_training_checkpoint
                    and model_logger.num_steps % save_steps == 0
                ):
                    save_training_checkpoint(
                        accelerator,
                        model_logger.output_path,
                        training_checkpoint_dir,
                        model_logger.num_steps,
                        epoch_id,
                        batch_id,
                        batches_per_epoch,
                        num_epochs,
                    )
                if eval_callback is not None and eval_callback.should_run(model_logger.num_steps):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        eval_callback(accelerator, model, model_logger)
                    accelerator.wait_for_everyone()
        start_batch_in_epoch = 0
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
