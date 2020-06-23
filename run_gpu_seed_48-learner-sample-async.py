from ray import tune
from ray.rllib.agents.seed import SeedTrainer
tune.run(SeedTrainer, config={"env": "BreakoutNoFrameskip-v4",
                              # Configs for Resources
                              "num_gpus": 0.5,
                              "num_workers": 1,
                              "num_envs_per_worker": 48,
                              "num_cpus_per_worker": 48,
                              "num_gpus_per_worker": 0.5,
                              "remote_env_poll_size": 32,
                              "remote_worker_envs": True,
                              "sample_async": True,
                              "rollout_fragment_length": 20,
                              "train_batch_size": 640,
                              "sgd_minibatch_size": 640,
                              "sample_max_steps": 640,
                              "num_cpus_for_driver": 10,
                              "learner_sample_async": True
                              # "eager": True,
                              # "log_level": "INFO" # for verbose
                            },
                            local_dir="/data/ray_results"
                            )  # "log_level": "INFO" for verbose,
# "eager": True for eager execution,
# "torch": True for PyTorch
