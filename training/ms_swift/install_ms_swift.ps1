# Install ms-swift in the current Python environment.
# Run this script from the project root after activating your target virtual environment.

python -m pip install --upgrade pip
python -m pip install "ms-swift" -U

# Optional sanity check. This prints the CLI help if installation succeeded.
swift --help
