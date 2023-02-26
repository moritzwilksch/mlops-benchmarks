import asyncio
import time

import aiohttp
from inference.run_bm import data_point
from loguru import logger

logger.add(sink="logs/logs.txt")


async def async_restapi_onnx(session: aiohttp.ClientSession, d: dict):
    async with session.post(
        "http://localhost:8000/predict-onnx",
        json=d,
        headers={"ContentType": "application/json"},
        timeout=10,
    ) as resp:
        return await resp.json()


async def async_restapi_joblib(session: aiohttp.ClientSession, d: dict):
    async with session.post(
        "http://localhost:8000/predict-joblib",
        json=d,
        headers={"ContentType": "application/json"},
        timeout=100,
    ) as resp:
        return await resp.json()


async def main():
    async with aiohttp.ClientSession() as session:
        futures = [
            asyncio.create_task(async_restapi_onnx(session=session, d=data_point))
            for _ in range(10_000)
        ]
        res = await asyncio.gather(*futures)
    print(res[:2])


start = time.perf_counter()
asyncio.run(main())
end = time.perf_counter()
logger.info(f"asyncio took {end - start:.4f} seconds.")
