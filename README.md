# WSM project 1

## Quick Start

1. Install Poetry
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Use Python 3.11 to run the project
   ```sh
   pyenv local 3.11  # if you have pyenv installed
   poetry env use $(command -v python3)  # make sure the right python is used
   ```
3. Install dependencies
   ```sh
   poetry install --no-root
   ```
4. Run the program
   ```sh
   poetry run python3 main.py

## Note

Since I'm using `ckiptagger`, it needs to download model files. Running the program will download them, but it will take a while.