from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceTFPolicy
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import LocalMultiGPUOptimizer, SyncSamplesOptimizer
from ray.rllib.utils.annotations import override
from ray.tune.trainable import Trainable
from ray.tune.resources import Resources

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # V-trace params (see vtrace.py).
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    # System params.
    #
    # == Overview of data flow in IMPALA ==
    # 1. Policy evaluation in parallel across `num_workers` actors produces
    #    batches of size `rollout_fragment_length * num_envs_per_worker`.
    # 2. If enabled, the replay buffer stores and produces batches of size
    #    `rollout_fragment_length * num_envs_per_worker`.
    # 3. If enabled, the minibatch ring buffer stores and replays batches of
    #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    #    on batches of size `train_batch_size`.
    #
    "rollout_fragment_length": 50,
    "train_batch_size": 500,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 500,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": False,
    "min_iter_time_s": 10,
    "num_workers": 0,
    # number of GPUs the learner should use.
    "num_gpus": 0,
    # Number of environments to evaluate vectorwise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    "num_envs_per_worker": 5,
    # number of passes to make over each train batch
    "num_sgd_iter": 1,

    # Learning params.
    "grad_clip": 40.0,
    # either "adam" or "rmsprop"
    "opt_type": "adam",
    "lr": 0.0005,
    "lr_schedule": None,
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,

    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # use fake (infinite speed) sampler for testing
    "_fake_sampler": False,
    "_fake_collect": False,
    "_fake_load_data": False,
    "_fake_update": False,
    "_fake_optimize": False,
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    "simple_optimizer": False,
    # Whether to fake GPUs (using CPUs).
    # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
    "_fake_gpus": False,
    # Use PyTorch as framework?
    "use_pytorch": False,

    "sample_max_steps": 0,
    "learner_sample_async": False
})
# __sphinx_doc_end__
# yapf: enable


def choose_policy(config):
    if config["vtrace"]:
        return VTraceTFPolicy
    else:
        return A3CTFPolicy


def validate_config(config):
    # PyTorch check.
    if config["use_pytorch"]:
        raise ValueError(
            "IMPALA does not support PyTorch yet! Use tf instead.")
    if config["entropy_coeff"] < 0:
        raise DeprecationWarning("entropy_coeff must be >= 0")

def choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=[])

    return LocalMultiGPUOptimizer(
        workers,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        rollout_fragment_length=config["rollout_fragment_length"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=[],
        shuffle_sequences=config["shuffle_sequences"],
        _fake_gpus=config["_fake_gpus"],
        sample_max_steps=config["sample_max_steps"],
        learner_sample_async=config["learner_sample_async"],
        _fake_collect=config["_fake_collect"],
        _fake_load_data=config["_fake_load_data"],
        _fake_optimize=config["_fake_optimize"])

SeedTrainer = build_trainer(
    name="SEED",
    default_config=DEFAULT_CONFIG,
    default_policy=VTraceTFPolicy,
    get_policy_class=choose_policy,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config)
