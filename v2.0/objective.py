import os
import subprocess
import optuna
from ruamel.yaml import YAML
from utils.YParams import YParams

class Objective:
    def __init__(self, config_file, run_num):
        self.config_file = config_file
        self.run_num = run_num

    def __call__(self, trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        drop_path = trial.suggest_float('drop_path', 0.0, 0.5)
        # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        # max_epochs = trial.suggest_int('max_epochs', 5, 50)
        scheduler = trial.suggest_categorical('scheduler', ['CosineAnnealingLR', 'ReduceLROnPlateau'])

        # Load base config
        params = YParams(self.config_file, 'PLASIM')

        # Update hyperparameters in config
        params['lr'] = lr
        params['weight_decay'] = weight_decay
        params['drop_path'] = drop_path
        # params['batch_size'] = batch_size
        # params['max_epochs'] = max_epochs
        params['scheduler'] = scheduler

        # Save the updated parameters to a temporary config file
        temp_config_file = f'/tmp/config_{self.run_num}.yaml'
        with open(temp_config_file, 'w') as file:
            yaml = YAML()
            yaml.dump({'PLASIM': params.params}, file)

        # Run the training script with the updated config
        cmd = [
            'python', '/scratch/user/u.pr160292/PanguWeather/v2.0/train.py',
            '--yaml_config', temp_config_file,
            '--config', 'PLASIM',
            '--run_num', str(self.run_num)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)  # Debugging output
        print(result.stderr)  # Debugging output

        # Parse the output to get the validation loss (example, modify as needed)
        validation_loss = self.parse_output(result.stdout)

        return validation_loss

    def parse_output(self, output):
        # Parse the output to find the validation loss
        for line in output.splitlines():
            if "Validation Epoch" in line:
                parts = line.split('Loss: ')
                if len(parts) > 1:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        return float('inf')
        return float('inf')

# Example of running the optimization
if __name__ == "__main__":
    config_file = '/scratch/user/u.pr160292/PanguWeather/v2.0/model_config.yaml'
    run_num = 1
    study = optuna.create_study(direction='minimize')
    objective = Objective(config_file, run_num)
    study.optimize(objective, n_trials=100)
    print(f'Best hyperparameters: {study.best_params}')

