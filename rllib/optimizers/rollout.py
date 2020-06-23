import logging

import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.memory import ray_get_and_free
import queue
import threading

logger = logging.getLogger(__name__)


def collect_samples(agents, rollout_fragment_length, num_envs_per_worker,
                    train_batch_size, sample_max_steps=0):
    """Collects at least train_batch_size samples, never discarding any."""

    num_timesteps_so_far = 0
    trajectories = []
    agent_dict = {}

    for agent in agents:
        fut_sample = agent.sample.remote()
        agent_dict[fut_sample] = agent

    while agent_dict:
        [fut_sample], _ = ray.wait(list(agent_dict))
        agent = agent_dict.pop(fut_sample)
        next_sample = ray_get_and_free(fut_sample)
        num_timesteps_so_far += next_sample.count
        trajectories.append(next_sample)

        # Only launch more tasks if we don't already have enough pending
        if sample_max_steps != 0:
            pending = len(agent_dict) * sample_max_steps
        else:
            pending = len(
                agent_dict) * rollout_fragment_length * num_envs_per_worker
        
        if num_timesteps_so_far + pending < train_batch_size:
            fut_sample2 = agent.sample.remote()
            agent_dict[fut_sample2] = agent

    return SampleBatch.concat_samples(trajectories)


# TODO: Use a async collector
class AsyncCollector(threading.Thread):
    def __init__(self, agents, rollout_fragment_length, num_envs_per_worker,
                    train_batch_size, sample_max_steps=0):
        threading.Thread.__init__(self)
        self.queue = queue.Queue()
        self.agents = agents
        self.train_batch_size = train_batch_size
        self.sample_max_steps = sample_max_steps

        self.pending = {a.sample.remote(): a for a in self.agents}
        self.shutdown = False
        
        if self.sample_max_steps:
            self.batch_bound = len(self.agents) * self.sample_max_steps
        else:
            self.batch_bound = len(self.agents) * rollout_fragment_length * num_envs_per_worker

    def run(self):
        try:
            self._run()
        except BaseException as e:
            self.queue.put(e)
            raise e

    def _run(self):
        sample_provider = self._collect()
        while not self.shutdown:
            item = next(sample_provider)
            self.queue.put(item)
    
    def _collect(self):
        num_timesteps_so_far = 0
        trajectories = []
        while not self.shutdown:
            [fut_sample], _ = ray.wait(list(self.pending))
            agent = self.pending.pop(fut_sample)
            next_sample = ray_get_and_free(fut_sample)
            num_timesteps_so_far += next_sample.count
            trajectories.append(next_sample)
            if num_timesteps_so_far >= self.batch_bound:
                yield SampleBatch.concat_samples(trajectories)
                num_timesteps_so_far = 0
                trajectories = []
            fut_sample2 = agent.sample.remote()
            self.pending[fut_sample2] = agent

    def collect_samples(self):
        if not self.is_alive():
            raise RuntimeError("Collecting thread has died")
        batch = self.queue.get(timeout=600.0)

        # Propagate errors
        if isinstance(batch, BaseException):
            raise batch

        return batch
            
