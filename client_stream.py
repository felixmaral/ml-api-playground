# client_stream.py
import asyncio, glob, json, os, time
import websockets

WS_URL = "ws://127.0.0.1:8000/ws"
IMG_DIR = "data"
PATTERN = "*.jpg"

async def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, PATTERN)))
    if not files:
        print("No images found in", IMG_DIR)
        return

    sent = 0
    t0 = time.time()
    async with websockets.connect(WS_URL, max_size=None) as ws:
        for path in files:
            with open(path, "rb") as f:
                raw = f.read()
            t1 = time.time()
            await ws.send(raw)
            msg = await ws.recv()
            dt = (time.time() - t1) * 1000
            out = json.loads(msg)
            print(f"{os.path.basename(path)} -> {out['pred_label']} "
                  f"(score={out['score']:.3f}, infer={out['time_ms']} ms, total={dt:.1f} ms)")
            sent += 1

    total = time.time() - t0
    print(f"\nProcessed {sent} images in {total:.2f}s "
          f"({sent/total:.2f} img/s, {1000*(total/sent):.1f} ms/img avg)")

if __name__ == "__main__":
    asyncio.run(main())