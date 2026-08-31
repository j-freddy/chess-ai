# Chess AI

Currently offline: https://lichess.org/@/MirroredBot

## Usage Guide

### Quick Start

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
as the package manager.

1. Clone this repository.

```sh
git clone https://github.com/j-freddy/chess-ai
cd chess-ai
```

2. Create virtual environment with Python 3.12+.

```sh
uv venv --python 3.12
```

3. Activate virtual environment.

```sh
source .venv/bin/activate
```

4. Install dependencies.

```sh
uv sync --no-dev
```

If everything has been set up correctly, you can now run the command below.
```sh
# Press Ctrl+C to exit
python -m main -white human -black human
# Or observe 2 bots play
python -m main -white airandom -black airandom
```

### Configuration

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

### Export UCI

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

## Development Guide

Read this section if you want to make changes or contribute to this repository.

### Installation

1. Go through [Quick Start](#quick-start) in the Usage Guide.

2. Install dev dependencies.

```sh
uv sync --dev
```

3. On Visual Studio Code, install the [Ruff
   Extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
   The settings in `.vscode/` configures Ruff to format and lint your code on
   save.

### Static Analysis

```sh
# Lint
ruff check --fix

# Type check
mypy .

# Format
ruff format
```

### Testing

```sh
coverage run -m pytest -sv -o log_cli=true && coverage report -m
```
