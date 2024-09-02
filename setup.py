import subprocess
import sys
import os

def install_dependencies():
    requirements = [
        "torch>=1.10.0",
        "numpy>=1.17.2",
        "scipy>=1.6.0",
        "pandas>=1.3.0",
        "tqdm>=4.48.2",
        "colorlog==4.7.2",
        "colorama==0.4.4",
        "scikit_learn>=0.23.2",
        "pyyaml>=5.1.0",
        "tensorboard>=2.5.0",
        "thop>=0.1.1.post2207130030",
        "tabulate>=0.8.10",
        "plotly>=4.0.0",
        "texttable>=0.9.0",
        "psutil>=5.9.0",
        "ray>=1.13.0,<=2.6.3",
    ]
    
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    install_dependencies()
    
    # Print package information as in setup.py
    print("RecBole version 1.2.0")
    print("A unified, comprehensive, and efficient recommendation library.")
    print("Developed based on Python and PyTorch for reproducing and developing recommendation algorithms.")
    print("For more information, visit: https://github.com/RUCAIBox/RecBole")
    
    # Add any other functionality you want to test
    print("Simulating package functionality...")

if __name__ == "__main__":
    main()
