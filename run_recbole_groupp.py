import argparse

def main():
    print("RecBole and its dependencies are not available. Running fallback mode.")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_list", "-m", type=str, default="BPR", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--valid_latex", type=str, default="./latex/valid.tex", help="output file for validation results"
    )
    parser.add_argument(
        "--test_latex", type=str, default="./latex/test.tex", help="output file for test results"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the IP of the master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of the master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args = parser.parse_args()

    # Placeholder for functionality previously relying on recbole
    model_list = args.model_list.strip().split(",")
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    valid_file = args.valid_latex.strip()
    test_file = args.test_latex.strip()

    print(f"Model List: {model_list}")
    print(f"Dataset: {args.dataset}")
    print(f"Config Files: {config_file_list}")
    print(f"Validation LaTeX File: {valid_file}")
    print(f"Test LaTeX File: {test_file}")
    print(f"Number of Processes: {args.nproc}")
    print(f"IP: {args.ip}")
    print(f"Port: {args.port}")
    print(f"World Size: {args.world_size}")
    print(f"Group Offset: {args.group_offset}")

    # Simulate placeholder results since recbole is not available
    valid_result_list = [{"Model": model, "Result": "Placeholder"} for model in model_list]
    test_result_list = [{"Model": model, "Result": "Placeholder"} for model in model_list]

    # Simulate writing results to LaTeX files
    with open(valid_file, "w") as f:
        f.write("Simulated validation results:\n")
        for result in valid_result_list:
            f.write(f"Model: {result['Model']}, Result: {result['Result']}\n")

    with open(test_file, "w") as f:
        f.write("Simulated test results:\n")
        for result in test_result_list:
            f.write(f"Model: {result['Model']}, Result: {result['Result']}\n")

    print("Fallback mode completed successfully.")

if __name__ == "__main__":
    main()
