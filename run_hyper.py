import argparse
import os
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import math

# Debug start of script
print("Script is running...")

# Mock objective function for demonstration
def objective_function(config):
    """
    A mock objective function that produces predictable results for demonstration purposes.
    """
    param1 = config.get("param1", 0.5)
    param2 = config.get("param2", 10)
    
    # Simulate a computation
    recall_at_10 = 0.9 - (param1 * 0.2) + (param2 * 0.01)
    
    result = {
        "recall@10": recall_at_10  # Compute the dummy result
    }

    # Print the result for each trial
    print(f"Trial with param1={param1}, param2={param2} resulted in recall@10={recall_at_10}")
    
    return result

def ray_tune(args):
    """
    Perform hyperparameter tuning using Ray Tune.
    """
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else []
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if config_file_list
        else []
    )
    params_file = (
        os.path.join(os.getcwd(), args.params_file) if args.params_file else None
    )
    
    ray.init(log_to_driver=True)  # Ensure Ray logs to console
    tune.register_trainable("train_func", objective_function)
    
    config = {}
    if params_file:
        with open(params_file, "r") as fp:
            for line in fp:
                para_list = line.strip().split(" ")
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = (
                    para_list[0],
                    para_list[1],
                    "".join(para_list[2:]),
                )
                if para_type == "choice":
                    para_value = eval(para_value)
                    config[para_name] = tune.choice(para_value)
                elif para_type == "uniform":
                    low, high = para_value.strip().split(",")
                    config[para_name] = tune.uniform(float(low), float(high))
                elif para_type == "quniform":
                    low, high, q = para_value.strip().split(",")
                    config[para_name] = tune.quniform(float(low), float(high), float(q))
                elif para_type == "loguniform":
                    low, high = para_value.strip().split(",")
                    config[para_name] = tune.loguniform(
                        math.exp(float(low)), math.exp(float(high))
                    )
                else:
                    raise ValueError(f"Illegal parameter type [{para_type}]")

    scheduler = ASHAScheduler(
        metric="recall@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    # Use an absolute path for storage_path
    storage_path = os.path.abspath("./ray_log")  # Updated to absolute path

    try:
        result = tune.run(
            tune.with_parameters(objective_function),
            config=config,
            num_samples=5,
            log_to_file=args.output_file,
            scheduler=scheduler,
            storage_path=storage_path,  # Updated argument
            resources_per_trial={"gpu": 0, "cpu": 1},  # Adjusted resources per trial
        )

        best_trial = result.get_best_trial("recall@10", "max", "last")

        # Print the best trial results
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final recall@10: {best_trial.last_result['recall@10']}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ray.shutdown()  # Ensure Ray shuts down properly

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", type=str, default=None, help="Config files")
    parser.add_argument("--params_file", type=str, default=None, help="Parameters file")
    parser.add_argument("--output_file", type=str, default=None, help="Output file")

    args = parser.parse_args()

    # Debug start of function
    print("Calling ray_tune function...")
    ray_tune(args)
    print("ray_tune function completed.")
