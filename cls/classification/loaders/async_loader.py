import aiohttp
import asyncio
import os
import logging
from os.path import join, exists

LOGGER = logging.getLogger(__name__)
URL = "https://static.yapics.collect.monster/"


class Counter:
    total_num: int = 0
    downloaded_num: int = 0
    failed_num: int = 0
    
    def __repr__(self) -> str:
        return f"[{self.downloaded_num + self.failed_num}/{self.total_num}]"
    
    
        
# async def download_file(session, url, images_dir):
#     async with session.get(url) as response:
#         if response.status == 200:
#             filename = url.split("/")[-1]
#             file_path = join(images_dir, filename)
#             if not exists(file_path):
#             #     pass
#             #     # print(f"File {filename} already exists")
#             # else:
#                 with open(file_path, "wb") as f:
#                     while True:
#                         chunk = await response.content.read(1024)
#                         if not chunk:
#                             break
#                         f.write(chunk)
#                 print(f"Downloaded {filename}")
#         else:
#             print(f"Failed to download {url}")


async def download_files(session, urls, images_dir, counter: Counter):
    for url in urls:
        url = url.replace('yapics', 'yapics2.dev')
        async with session.get(url) as response:
            if response.status != 200:
                counter.failed_num += 1
                LOGGER.info(f"{counter} Failed to download {url}. Status - {response.status}")
                continue
            
            filename = url.split("/")[-1]
            file_path = join(images_dir, filename)
            if exists(file_path):
                counter.downloaded_num += 1
                LOGGER.info(f"{counter} {filename} already exists")
                continue
            
            ret = await save_content(response, file_path)
            if ret is False:
                counter.failed_num += 1
                LOGGER.info(f"{counter} Failed to save {url}.")
                continue
            
            counter.downloaded_num += 1
            LOGGER.info(f"{counter} Downloaded {filename}")


async def save_content(response: aiohttp.ClientResponse, file_path: str, chunk_size=1024) -> bool:    
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
                f.write(chunk)
        
        return True
    
    except Exception as e:
        print(e)
    
    return False
            

async def download_images(urls: list, images_dir: str, batch_size: int = 50):
    os.makedirs(images_dir, exist_ok=True)
    
    full_urls = urls
    
    # for url in urls:
    #     filename = url.split("/")[-1]
    #     file_path = os.path.join(images_dir, filename)
    #     if os.path.exists(file_path):
    #         continue
        
    #     full_url = os.path.join(URL, url)
    #     full_urls.append(full_url)
    
    url_batches = []    
    for i in range(0, len(full_urls), batch_size):
        url_batches.append(full_urls[i: min(i + batch_size, len(full_urls))])
    
    counter = Counter()
    counter.total_num = len(full_urls)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url_batch in url_batches:
            tasks.append(download_files(session, url_batch, images_dir, counter))

        await asyncio.gather(*tasks)
    
    LOGGER.info(f"Result: downloaded - {counter.downloaded_num}, failed - {counter.failed_num}")
    

# urls = [
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/6a4b9fa464a67a2c7b6cb21fbccb6f6f.jpeg",
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/72ef25b872b963c5973f9d61c40fc662.jpeg",
#     "https://static.yapics.dev.collect.monster/d/9/e/d9eff467-cb2e-4f9f-8a53-677d729d7a16/p/a2dda55bdbf073c5533a2728e37cb1c1.jpeg",
# ]

# asyncio.run(main(urls))
