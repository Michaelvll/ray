from ray.rllib.agents.seed.seed import SeedTrainer, DEFAULT_CONFIG
from ray.rllib.utils import renamed_agent

ImpalaAgent = renamed_agent(SeedTrainer)

__all__ = ["SeedAgent", "SeedTrainer", "DEFAULT_CONFIG"]
