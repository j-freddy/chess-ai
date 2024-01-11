"""
Start a server with WebSockets to communicate with LiChess API

Test locally by starting this server and running the command on another
terminal:
python -m websockets ws://localhost:8000/
"""

import asyncio
import websockets
from websockets.server import serve

async def handler(websocket):
    # Accept infinite incoming messages from the client
    while True:
        try:
            message = await websocket.recv()
            print(f"Received message: {message}")
            print("Sending echo...")
            await websocket.send(f"Echo: {message}")
        except websockets.ConnectionClosedOK:
            print("Connection closed")
            break

async def main():
    print("Starting server...")

    async with serve(handler, "localhost", 8000):
        # Run forever
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
