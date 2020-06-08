from ray import tune
from ray.rllib.agents.seed import SeedTrainer
tune.run(SeedTrainer, config={"env": "BreakoutNoFrameskip-v4",
                              # Configs for Resources
                              "num_gpus": 0.5,
                              "num_workers": 1,
                              "num_envs_per_worker": 10,
                              "num_cpus_per_worker": 10,
                              "num_gpus_per_worker": 0.5,
                              "remote_env_poll_size": 5,
                              "remote_worker_envs": True,
                              "sample_async": True,
                              # "eager": True,
                            #   "log_level": "INFO" # for verbose
                            },
                            local_dir="/data/ray_results"
                            )  # "log_level": "INFO" for verbose,
# "eager": True for eager execution,
# "torch": True for PyTorch
