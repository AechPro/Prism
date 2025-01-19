import torch
import wandb


class Logger(object):
    def __init__(self, config=None, holdout_data=None):
        self.wandb_run = None
        if config is not None:
            if config.log_to_wandb:
                project_name = config.wandb_project_name
                group_name = config.wandb_group_name
                run_name = config.wandb_run_name
                if run_name == "null":
                    run_name = config.env_name

                self.wandb_run = wandb.init(project=project_name, group=group_name, name=run_name, reinit=True)
                self.wandb_run.config.update(config)

        self.current_data = {}
        self.current_iteration = 0
        self.hidden_keys = []
        self.holdout_data = holdout_data
        self.enabled = True

    def set_holdout_data(self, data):
        self.holdout_data = data

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def override_log(self, data, group_name, var_name):
        self.log_data(data, group_name, var_name, override_enable=True)

    def log(self, data, group_name, var_name):
        self.log_data(data, group_name, var_name)

    def log_data(self, data, group_name, var_name, override_enable=False):
        if override_enable or self.enabled:
            name = "{}-{}".format(group_name, var_name)
            self.current_data[name] = data
            if self.wandb_run is not None:
                self.wandb_run.log({name: data}, commit=False)

    def report(self, iteration=None, float_sig_figs=6):
        if iteration is None:
            iteration = self.current_iteration

        if self.wandb_run is not None:
            self.wandb_run.log({}, commit=True)

        delim = "*" * 20
        print("\n{} BEGIN ITERATION {} REPORT {}".format(delim, iteration, delim))
        for key, val in self.current_data.items():
            if isinstance(val, torch.Tensor):
                if val.numel == 1:
                    self.current_data[key] = val.item()

            if isinstance(val, float):
                self.current_data[key] = round(val, float_sig_figs)

        self._print_data()
        print("\n{} END ITERATION {} REPORT {}".format(delim, iteration, delim))
        self.current_iteration += 1

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _print_data(self):
        data = self.current_data

        # These two functions were written by Claude 3.5 Sonnet after a fair amount of massaging the prompt and giving
        # it feedback about failures.
        tree = self._build_tree(data)
        self._print_branch(tree)

    def _build_tree(self, data):
        tree = {}
        for key, value in data.items():
            parts = key.split('/')
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            last_part, var_name = parts[-1].split('-')
            if last_part not in current:
                current[last_part] = {}
            current[last_part][var_name] = value
        return tree

    def _print_branch(self, branch, indent=""):
        for key in sorted(branch.keys()):
            if key in self.hidden_keys:
                continue

            print(f"\n{indent}-- {key} --")
            deferred = []
            if isinstance(branch[key], dict):
                next_indent = indent + "  "
                for var_name, value in sorted(branch[key].items()):
                    if isinstance(value, dict):
                        deferred.append(({var_name: value}, next_indent))
                    else:
                        print(f"{next_indent}{var_name}:   {value}")
            for arg in deferred:
                self._print_branch(*arg)
