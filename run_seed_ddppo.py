from ray import tune
from ray.rllib.agents.ppo import DDPPOTrainer
tune.run(DDPPOTrainer, config={"env": "BreakoutNoFrameskip-v4",
                              "num_gpus": 0,
                              "num_workers": 1,
                              "num_envs_per_worker": 4,
                              "num_cpus_per_worker": 4,
                              "num_gpus_per_worker": 0,
                              "remote_worker_envs": True
                              })  # "log_level": "INFO" for verbose,
# "eager": True for eager execution,
# "torch": True for PyTorch
