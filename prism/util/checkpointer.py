import os
import shutil
import numpy as np
import torch
import time


class Checkpointer(object):
    def __init__(self, save_dir, agent, experience_buffer, timesteps_per_agent_checkpoint, hours_per_backup):
        self.save_dir = save_dir
        self.agent = agent
        self.experience_buffer = experience_buffer
        self.timesteps_per_agent_checkpoint = timesteps_per_agent_checkpoint
        self.last_backup_checkpoint_time = time.time()
        self.last_agent_checkpoint_timesteps = 0
        self.seconds_per_backup = hours_per_backup*60*60

    def load_checkpoint(self, checkpoint_dir):
        # Load both agent and experience buffer
        self.agent.load(checkpoint_dir)
        self.experience_buffer.load(checkpoint_dir)

    def save_backup_checkpoint(self):
        return

        # Save both agent and experience buffer
        backup_checkpoint_path = os.path.join(self.save_dir, "backup_checkpoint")
        self.agent.save(backup_checkpoint_path)
        self.experience_buffer.save(backup_checkpoint_path)

    def save_agent_checkpoint(self):
        # Save only the agent
        agent_checkpoint_path = os.path.join(self.save_dir, f"agent_checkpoint_{self.last_agent_checkpoint_timesteps}")
        self.agent.save(agent_checkpoint_path)

    def checkpoint(self, timesteps):
        # Save backup checkpoint every hour
        current_time = time.time()
        if current_time - self.last_backup_checkpoint_time > self.seconds_per_backup:
            print("Saving backup checkpoint at timestep {}. It has been {} seconds since last backup.".
                  format(timesteps, current_time - self.last_backup_checkpoint_time))

            self.save_backup_checkpoint()
            self.last_backup_checkpoint_time = current_time

        # Save agent checkpoint every timesteps_per_agent timesteps
        if timesteps - self.last_agent_checkpoint_timesteps >= self.timesteps_per_agent_checkpoint:
            print("Saving agent checkpoint at timestep {}".format(timesteps))

            self.last_agent_checkpoint_timesteps = timesteps
            self.save_agent_checkpoint()


