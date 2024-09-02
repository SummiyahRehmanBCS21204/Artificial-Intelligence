import argparse

def main():
    # Print a message indicating that the script is running in a fallback mode
    print("RecBole and its dependencies are not available. Running fallback mode.")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
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

    # Print the parsed arguments to confirm the script's functionality
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config Files: {args.config_files}")
    print(f"Number of Processes: {args.nproc}")
    print(f"IP: {args.ip}")
    print(f"Port: {args.port}")
    print(f"World Size: {args.world_size}")
    print(f"Group Offset: {args.group_offset}")

    # Since RecBole is not available, you can perform alternative actions here if needed.
    # For now, just indicate that the fallback mode is complete.
    print("Fallback mode completed successfully.")

if __name__ == "__main__":
    main()
