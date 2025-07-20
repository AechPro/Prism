from prism import Learner
import time


class ExperimentRunner(object):
    def __init__(self):
        self.learner = Learner()
        self.experiments = []

    def register_experiment(self, experiment):
        self.experiments.append(experiment)

    def run_experiments(self):
        for experiment in self.experiments:
            print("BEGINNING EXPERIMENT", experiment)
            print("NUMBER OF CONFIGS TO RUN:", len(experiment.configs))
            start_time = time.time()
            while not experiment.is_done():
                cfg = experiment.get_next_config()
                self.learner.configure(cfg)

                config_start_time = time.time()
                self.learner.learn()
                print("CONFIG COMPLETE AFTER {:7.2f} MINUTES".format((time.time() - config_start_time) / 60))

            print("EXPERIMENT COMPLETE AFTER {:7.2f} MINUTES".format((time.time() - start_time) / 60))
