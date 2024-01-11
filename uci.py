"""
Implement the UCI protocol as publiced by Stefan-Meyer Kahlen (ShredderChess)

See uci-protocol.txt. Summary below.

GUI to engine:
    [DONE] uci
    [SKIP] debug [ on | off ]
    isready
    [SKIP] setoption name  [value ]
    [SKIP] register
    [DONE] ucinewgame
    position [fen  | startpos ]  moves  ....
    go ....
    stop
    ponderhit
    quit

Engine to GUI:
    [DONE] id name author
    [DONE] uciok
    [DONE] readyok
    bestmove  [ ponder  ]
    copyprotection
    registration
    info ....
    option ....
"""

def service_uci_command(command: str):
    if command == "uci":
        # https://lichess.org/@/MirroredBot
        print("id name MirroredBot")
        print("id author Freddy Jiang")
        print("uciok")
        return

    if command == "isready":
        print("readyok")
        return

    if command == "ucinewgame":
        return

    
        
            

if __name__=="__main__":
    while True:
        service_uci_command(input())
