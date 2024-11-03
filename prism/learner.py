from prism.factory import algorithm_factory
import time
import numpy as np


class Learner(object):
    def __init__(self):
        self.agent = None
        self.timestep_collector = None
        self.experience_buffer = None
        self.logger = None
        self.checkpointer = None
        self.timestep_limit = None
        self.initial_random_timesteps = None
        self.timesteps_per_iteration = None
        self.timesteps_per_report = None
        self.target_network_update_period = None
        self.use_target_network = None
        self.use_per = None
        self.per_beta = None
        self.cumulative_timesteps = 0
        self.cumulative_model_updates = 0
        self.timesteps_since_report = 0
        self.timesteps_since_target_model_update = 0
        self.loggables = {}
        self.device = None

    def configure(self, config):
        (self.agent,
         self.timestep_collector,
         self.experience_buffer,
         self.logger,
         self.checkpointer) = algorithm_factory.build_algorithm(config)

        # Misc numbers
        self.timestep_limit = config.timestep_limit
        self.initial_random_timesteps = config.num_initial_random_timesteps

        # Periods
        self.timesteps_per_iteration = config.timesteps_per_iteration
        self.timesteps_per_report = config.timesteps_per_report
        self.target_network_update_period = config.target_update_period

        # `Use` flags
        self.use_target_network = config.use_target_network
        self.use_per = config.use_per
        if self.use_per:
            from prism.util.annealing_strategies import LinearAnneal
            self.per_beta = LinearAnneal(config.per_beta_start, config.per_beta_end, config.per_beta_anneal_timesteps)

        # Counters
        self.cumulative_timesteps = 0
        self.cumulative_model_updates = 0
        self.timesteps_since_report = 0
        self.timesteps_since_target_model_update = 0
        self.loggables = {}
        self.reset_loggables()
        self.device = config.device

    def _learn(self):
        self.cumulative_timesteps = self.timestep_collector.collect_timesteps(self.initial_random_timesteps,
                                                                              self.agent,
                                                                              self.experience_buffer,
                                                                              random=True)
        self.timesteps_since_report = self.cumulative_timesteps
        self.timetimesteps_since_target_model_update = self.cumulative_timesteps
        self.logger.set_holdout_data(self.experience_buffer.sample().clone())
        self.checkpointer.checkpoint(self.cumulative_timesteps)

        # from cProfile import Profile
        # profile = Profile()
        # profile.enable()

        while self.cumulative_timesteps < self.timestep_limit:
            loop_start = time.perf_counter()

            ### COLLECT TIMESTEPS ###
            t1 = time.perf_counter()
            timesteps_this_iteration = self.timestep_collector.collect_timesteps(self.timesteps_per_iteration,
                                                                                 self.agent,
                                                                                 self.experience_buffer)

            self.loggables["Timestep Collection Time"].append(time.perf_counter() - t1)
            self.cumulative_timesteps += timesteps_this_iteration
            self.timesteps_since_report += timesteps_this_iteration
            self.timesteps_since_target_model_update += timesteps_this_iteration

            ### SAMPLE BATCH ###
            if self.agent._static_batch is not None and self.cumulative_model_updates == 1:
                self.experience_buffer._batch = self.agent._static_batch

            t1 = time.perf_counter()
            batch, info = self.experience_buffer.sample(return_info=True)
            self.loggables["Batch Sampling Time"].append(time.perf_counter() - t1)

            if self.use_per:
                # beta = self.per_beta.update(timesteps_this_iteration)
                beta = 0.5
                per_weights = info['_weight'].to(self.device, non_blocking=True)
                self.experience_buffer.buffer._sampler._beta = beta
            else:
                per_weights = 1

            ### UPDATE AGENT ###
            t1 = time.perf_counter()
            new_per_weights = self.agent.update(batch, per_weights=per_weights)
            self.loggables["Agent Update Time"].append(time.perf_counter() - t1)
            self.cumulative_model_updates += 1

            ### OPTIONAL COMPONENT UPDATES ###
            t1 = time.perf_counter()
            if self.use_per:
                self.experience_buffer.update_priority(info['index'], new_per_weights.abs())

            if self.use_target_network and self.timesteps_since_target_model_update >= self.target_network_update_period:
                self.agent.sync_target_model()
                self.timesteps_since_target_model_update = 0
            self.loggables["Component Update Time"].append(time.perf_counter() - t1)

            ### CHECKPOINT AND REPORT ###
            self.checkpointer.checkpoint(self.cumulative_timesteps)
            if self.timesteps_since_report >= self.timesteps_per_report:
                self.report()
                # profile.disable()
                # profile.dump_stats("profile.prof")
                # profile.enable()

            self.loggables["Iteration Time"].append(time.perf_counter() - loop_start)

    def report(self):
        self._log()
        self.logger.report()

        self.timesteps_since_report = 0
        self.reset_loggables()

    def reset_loggables(self):
        self.loggables.clear()
        self.loggables["Component Update Time"] = []
        self.loggables["Iteration Time"] = []
        self.loggables["Agent Update Time"] = []
        self.loggables["Batch Sampling Time"] = []
        self.loggables["Iteration Time"] = []
        self.loggables["Timestep Collection Time"] = []

    def _log(self):
        self.agent.log(self.logger)
        self.timestep_collector.log(self.logger)
        if len(self.loggables["Iteration Time"]) == 0:
            return

        ts_col_time = sum(self.loggables["Timestep Collection Time"])
        iter_time = sum(self.loggables["Iteration Time"])
        collected_steps_per_second = self.timesteps_since_report / ts_col_time
        overall_steps_per_second = self.timesteps_since_report / iter_time

        if self.use_per:
            self.logger.log_data(data=self.per_beta.get_value(),
                                 group_name="Report/PER",
                                 var_name="Beta")

        self.logger.log_data(data=self.cumulative_timesteps,
                             group_name="Report/Metrics",
                             var_name="Cumulative Timesteps")

        self.logger.log_data(data=self.cumulative_model_updates,
                             group_name="Report/Model",
                             var_name="Number of Updates")

        self.logger.log_data(data=collected_steps_per_second,
                             group_name="Report/Metrics",
                             var_name="Collected Steps per Second")

        self.logger.log_data(data=overall_steps_per_second,
                             group_name="Report/Metrics",
                             var_name="Overall Steps per Second")

        for key, value in self.loggables.items():
            self.logger.log_data(data=np.mean(value),
                                 group_name="Report/Metrics",
                                 var_name=key)

    def learn(self):
        if self.agent is None:
            print("YOU MUST CONFIGURE THE LEARNER BEFORE CALLING LEARN!")
        else:
            try:
                self._learn()
            finally:
                print("SAVING EXIT CHECKPOINT")
                self.checkpointer.save_backup_checkpoint()

                print("CLEARING EXPERIENCE BUFFER")
                self.experience_buffer.empty()

                print("SHUTTING DOWN TIMESTEP COLLECTORS")
                self.timestep_collector.close()

                print("FINALIZING LOGGER")
                self.logger.close()

                print("LEARNER EXIT COMPLETE")
