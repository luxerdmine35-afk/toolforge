import uvicorn.protocols.http.h11_impl
import uvicorn.protocols.http.auto
import uvicorn.lifespan.on
import uvicorn.loops.auto
import asyncio
import random
import time
import re
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn
from PIL import Image
import imagehash
import json

MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
KEYFILE = "gemini.txt"
PHASH_API = "https://lib-gem-earnzilla.vercel.app/api/phash"
PROMPT_TEMPLATE = "\nPerhatikan gambar di atas. Gambar tersebut dibagi menjadi 6 grid (3 di atas, 3 di bawah, dibaca dari kiri ke kanan lalu dari atas ke bawah).\nTugas kamu adalah membaca instruksi di bagian atas, lalu cari nomor grid (dari angka 1 sampai 6) yang sesuai dengan urutan instruksi tersebut.\nAturan Penting:\n\nHanya ada 6 kotak grid, jadi jangan pernah menyebutkan angka di atas 6 (seperti 9).\nFormat jawaban wajib berupa kurung dan angka koma, contoh: (x, y, z).\nBerikan jawaban yang akurat sesuai urutan instruksinya!, dan jawab dengan ringkas , langsung sebut jawaban\n"
CONFIG_FILE = "config.json"


def load_config():
    try:
        with open(CONFIG_FILE, encoding="utf8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to load {CONFIG_FILE}: {e}")


config = load_config()
HOST = config.get("host", "127.0.0.1")
PORT = int(config.get("port", 8000))


class Req(BaseModel):
    image: str


class KeyPool:
    def __init__(self):
        self.keys = []
        self.busy = set()
        self.cool = {}
        self.cond = asyncio.Condition()
        self.load()

    def load(self):
        try:
            self.keys = [l.strip() for l in open(KEYFILE, encoding="utf8") if l.strip()]
        except FileNotFoundError:
            self.keys = []

    async def acquire(self):
        async with self.cond:
            while True:
                now = time.time()
                self.cool = {k: v for k, v in self.cool.items() if v > now}
                idle = [k for k in self.keys if k not in self.busy and k not in self.cool]
                if idle:
                    k = random.choice(idle)
                    self.busy.add(k)
                    return k
                await self.cond.wait()

    async def release(self, k):
        async with self.cond:
            self.busy.discard(k)
            self.cond.notify()

    async def cooldown(self, k, s=60):
        self.cool[k] = time.time() + s


pool = KeyPool()
client = httpx.AsyncClient(timeout=120)
app = FastAPI()


def compute_phash(image_b64: str) -> str:
    try:
        data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(data))
        return str(imagehash.phash(img))
    except Exception as e:
        raise HTTPException(400, f"Gagal decode/hash gambar: {e}")


async def get_cached_numbers(phash: str) -> list[int] | None:
    r = await client.get(PHASH_API)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and phash in data:
        nums_str = data[phash]
        return [int(x) for x in nums_str.split(",") if x.strip() != ""]
    return None


async def save_phash(phash: str, numbers: list[int]):
    payload = [{"phash": phash, "numbers": numbers}]
    r = await client.post(PHASH_API, json=payload)
    r.raise_for_status()


def parse_numbers_from_gemini(result: dict) -> list[int]:
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(500, "Format response Gemini gak sesuai ekspektasi")
    numbers = [int(n) for n in re.findall("-?\\d+", text)]
    if not numbers:
        raise HTTPException(500, "Gak ada angka yang ketemu di response Gemini")
    return numbers


async def call_gemini(image_b64: str) -> list[int]:
    if not pool.keys:
        raise HTTPException(500, "No keys in gemini.txt")
    payload = {
        "contents": [
            {"parts": [{"text": PROMPT_TEMPLATE}, {"inline_data": {"mime_type": "image/png", "data": image_b64}}]}
        ]
    }
    tried = set()
    while len(tried) < len(pool.keys):
        key = await pool.acquire()
        tried.add(key)
        try:
            r = await client.post(GEMINI_URL, params={"key": key}, json=payload)
            if r.status_code == 429:
                await pool.cooldown(key)
                continue
            r.raise_for_status()
            return parse_numbers_from_gemini(r.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, e.response.text)
        finally:
            await pool.release(key)
    raise HTTPException(429, "All keys are cooling down")


@app.post("/generate")
async def generate(req: Req):
    phash = compute_phash(req.image)
    cached = await get_cached_numbers(phash)
    if cached is not None:
        return cached
    numbers = await call_gemini(req.image)
    await save_phash(phash, numbers)
    return numbers


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
