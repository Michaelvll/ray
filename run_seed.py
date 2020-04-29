from ray import tune
from ray.rllib.agents.seed import SeedTrainer
tune.run(SeedTrainer, config={"env": "BreakoutNoFrameskip-v4",
                                "num_gpus": 0,
                                "num_workers": 0,
                                "num_envs_per_worker": 4
                                })  # "log_level": "INFO" for verbose,
# "eager": True for eager execution,
# "torch": True for PyTorch
