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
usage: main.py [-h] [-white WHITE] [-black BLACK] [-startpos STARTPOS]

options:
  -h, --help          show this help message and exit
  -white WHITE        White player. Options: human, airandom, aimcts
  -black BLACK        Black player. Options: human, airandom, aimcts
  -startpos STARTPOS  Starting position in FEN. Default: standard position.
```

For example, to play White against a smart AI, run the command below.
```sh
python -m main -white human -black aimcts
```

### Advanced

`uci.py` implements a bare-bones UCI protocol. For example, it can be used to
connect to a LiChess bot.

This command creates an executable file for the AI engine.
```
pyinstaller -F uci.py
```

The file is located in `dist/uci/uci.exe`.


Alternatively, you can also run the UCI engine directly.
```sh
python -m uci
```

## Contribute

### Test

```sh
# Run tests
pytest
# Run tests with coverage report
pytest --cov
```

### Update Dependencies

```sh
pip freeze > requirements.txt
```
