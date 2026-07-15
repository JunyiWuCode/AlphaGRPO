import importlib.util
import os


_BASE_PATH = os.path.join(os.path.dirname(__file__), "base_awm.py")
_SPEC = importlib.util.spec_from_file_location("base_awm", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


def get_config():
    """SpectraReward paper setup scaled from 32 to 8 policy GPUs."""
    config = _BASE.get_config()

    root = os.environ.get("ALPHAGRPO_ROOT", os.getcwd())
    config.run_name = os.environ.get("RUN_NAME", "spectrareward_sd35_qwen3vl8b")
    config.logdir = os.environ.get("LOG_DIR", os.path.join(root, "logs", "sd35"))
    config.save_dir = config.logdir
    config.resume_from = os.environ.get("RESUME_FROM", "")
    config.dataset = os.environ.get(
        "ALPHAGRPO20K_DIR",
        os.path.join(root, "alpha_grpo", "dataset", "alphagrpo20k"),
    )
    config.prompt_fn = "alphagrpo20k"
    config.pretrained.model = os.environ.get(
        "SD35_MODEL", "stabilityai/stable-diffusion-3.5-medium"
    )

    config.resolution = 512
    config.mixed_precision = "bf16"
    config.num_epochs = 380
    config.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", "380"))
    config.max_run_seconds = int(os.environ.get("MAX_RUN_SECONDS", "0"))
    config.skip_initial_epochs = 0
    config.save_freq = int(os.environ.get("SAVE_FREQ", "10"))
    config.eval_freq = int(os.environ.get("EVAL_FREQ", "1000000"))
    config.use_lora = True
    config.lora_type = "attn"

    config.sample.num_steps = 16
    config.sample.eval_num_steps = 16
    config.sample.guidance_scale = 4.0
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 2
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 32
    config.sample.noise_level = 0.0
    config.sample.sde_frac = 0.0
    config.sample.global_std = True

    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 32
    config.train.num_batches_per_epoch = 32
    config.train.learning_rate = 1e-4
    config.train.train_timesteps = 6
    config.train.timestep_fraction = 10 / 16
    # CFG=4 is used for rollouts; official SD3 AWM trains the conditional field.
    config.train.cfg = False
    config.train.beta = 0.001
    config.train.kl_weight = "Uniform"
    config.train.adv_clip_max = 5
    config.train.advantage_max = 1
    config.train.clip_range = 1.0
    config.train.loss_type = "exp_first"
    config.train.weighting = "ghuber"
    config.train.ghuber_power = 0.25
    config.train.off_policy = False
    config.train.use_sa_solver = True
    config.train.decay_steps = 10**10
    config.train.ema = True
    config.train.ema_decay = 0.99
    config.train.ema_update_step_interval = 1
    config.train.ema_beta = 1.0
    config.train.kl_ema_weight = "Uniform"
    config.train.kl_ema_decay = 0.3
    config.train.kl_ema_decay_type = "linear"

    config.time_type = "discrete_wo_init"
    config.time_shift = 3.0
    config.reward_fn = {"spectrareward": 1.0}
    config.per_prompt_stat_tracking = True
    return config
