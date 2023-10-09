import aiohttp
import asyncio
import os
from os.path import join, exists

#DATA = "/home/achernikov/DATA/new_bg/picture"
URL = "https://static.yapics.collect.monster/"


async def download_file(session, url, images_dir):
    async with session.get(url) as response:
        if response.status == 200:
            filename = url.split("/")[-1]
            file_path = join(images_dir, filename)
            if not exists(file_path):
            #     pass
            #     # print(f"File {filename} already exists")
            # else:
                with open(file_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {url}")


async def download_images(urls, images_dir: str):
    os.makedirs(images_dir, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(download_file(session, join(URL, url), images_dir))

        await asyncio.gather(*tasks)


# urls = [
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/6a4b9fa464a67a2c7b6cb21fbccb6f6f.jpeg",
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/72ef25b872b963c5973f9d61c40fc662.jpeg",
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/a2dda55bdbf073c5533a2728e37cb1c1.jpeg",
# ]

# asyncio.run(main(urls))