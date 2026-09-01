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

4. Install the package and its dependencies.

```sh
uv sync --no-dev
```

If everything has been set up correctly, you can now run the command below.
```sh
# Press Ctrl+C to exit
chess-ai -white human -black human
# Or observe 2 bots play
chess-ai -white airandom -black airandom
```

Equivalently, without the installed script:
```sh
python -m chess_ai -white airandom -black airandom
```

### Configuration

```sh
usage: chess-ai [-h] -white WHITE -black BLACK [-startpos STARTPOS]

options:
  -h, --help          show this help message and exit
  -white WHITE        White player. Options: human, airandom, aimcts
  -black BLACK        Black player. Options: human, airandom, aimcts
  -startpos STARTPOS  Starting position in FEN. Default: standard position.
```

For example, to play White against a smart AI, run the command below.
```sh
chess-ai -white human -black aimcts
```

### Export UCI

`chess_ai.uci` implements a bare-bones UCI protocol. For example, it can be used
to connect to a LiChess bot.

Run the UCI engine directly:
```sh
chess-ai-uci
# Or: python -m chess_ai.uci
```

To build a standalone executable for the engine:
```sh
pyinstaller -F -n chess-ai-uci --paths src src/chess_ai/uci/__main__.py
```

The file is located in `dist/chess-ai-uci`.

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

### Project Layout

```
src/chess_ai/
    cli.py              Command line entry point
    game.py             Game loop and move validation
    chess_types.py      State and Action aliases shared across the package
    players/            Player abstraction and its implementations
        base.py         Player: chooses a legal move for a position
        human.py        Human: reads and validates moves from stdin
        ai.py           AI: a Player that identifies itself and has a budget
        ai_random.py    AIRandom
        mcts/           AIMCTS and its Monte Carlo search tree
    models/             Position evaluators used by AIMCTS
    uci/                UCI protocol implementation and engine entry point
tests/                  Mirrors the package
```

A `Player` promises to return a move that is legal in the position it is given.
`Human` upholds that promise by re-prompting until the input parses to a legal
move. `Game` verifies the promise regardless and raises `IllegalMoveError` if a
player breaks it, so a faulty engine stops the game with a clear report rather
than a traceback.

All tool configuration lives in `pyproject.toml`.

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
