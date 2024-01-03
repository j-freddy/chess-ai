# Chess AI

## Usage Guide

### Installation

1. Clone this repository.

```sh
git clone https://github.com/j-freddy/chess-ai
```

2. Create virtual environment with Python 3.10+.

```sh
# Go inside repo
cd chess-ai
# Check Python 3.10+ is used
python --version
# Create virtual environment
python -m venv venv
# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies.

```sh
pip install -r requirements.txt
```

To check everything is set up correctly, try playing a game.
```sh
# Press Ctrl+C to exit
python -m main -white human -black human
# Or observe 2 bots play
python -m main -white airandom -black airandom
```

### Usage

```sh
usage: main.py [-h] [-white WHITE] [-black BLACK]

options:
  -h, --help    show this help message and exit
  -white WHITE  White player. Options: human, airandom, aimcts
  -black BLACK  Black player. Options: human, airandom, aimcts
```

For example, to play White against a smart AI, run the command below.
```sh
python -m main -white human -black aimcts
```

## Contribute

### Update Dependencies

```sh
pip freeze > requirements.txt
```
