import os
import sys
import json
import asyncio
import aiohttp
import requests
from cls.classification.loaders.meta_async_loader import fetch_meta


class YapicsAPI:
    def __init__(self, 
                 stand: str = 'dev.', 
                 base_url: str = "https://yapics2.{stand}collect.monster/v1",
        ) -> None:

        self.base_url = base_url.format(stand=stand)
    
    def get_token(self):
        headers = {"Content-Type": "application/json"}
        data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}
        url = f"{self.base_url}/login"

        r = requests.post(url, data=json.dumps(data_log), headers=headers)
        token = eval(r.text)["token"]
        return token

    def load_meta(self, token: str, picset_ids):
        url = f"{self.base_url}/meta/picsets"
        head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

        guids = {"guids": picset_ids}
        r1 = requests.post(url, data=json.dumps(guids), headers=head, timeout=500000)
        return r1
    
    def get_data(self, group: str, token: str):
        url = f"{self.base_url}/meta/pictures"
        head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

        groups = {"listBy": 0, "categories": [], "groups": [group], "mode" : ["PREPARING","CHECKING","VALIDATED"]}
        r1 = requests.post(url, data=json.dumps(groups), headers=head, timeout=500000)
        data = r1.json()["data"]
        
        return data
    
    def post_trained(self, token, data):
        url = f"{self.base_url}/picset/trained"
        head = {"Authorization": f"token {token}"}
        r1 = requests.post(url, data=json.dumps(data), headers=head, timeout=500000)
        return r1
    
    def set_checking(self, token: str, js_path: str):
        head = {"Authorization": f"token {token}"}
        url = f"{self.base_url}/picset/checking"
        r1 = requests.post(
            url,
            data=json.dumps({"guids": [js_path.split("/")[-2]]}),
            headers=head,
            timeout=500000,
        )
        return r1

    def get_downloading_urls(self, urls: list, images_dir: str):
        prefix_url = "https://static.yapics.collect.monster/"
        full_urls = []
        
        for url in urls:
            filename = url.split("/")[-1]
            file_path = os.path.join(images_dir, filename)
            if os.path.exists(file_path):
                continue
            
            full_url = os.path.join(prefix_url, url)
            full_urls.append(full_url)
        
        return full_urls
    
    async def get_meta(self, bearer, guids, group, meta_dir):
        url = f'{self.base_url}/meta/picsets'
        head = {"Authorization": f"bearer {bearer}", "Content-Type": "application/json"}
        
        async with aiohttp.ClientSession(trust_env = True) as session:
            tasks = [fetch_meta(session, url, head, guid, group, meta_dir) for guid in guids["guids"]]
            await asyncio.gather(*tasks)
    
    
api = YapicsAPI()
token = api.get_token()
print(token)
# url = f"https://yapics2.dev.collect.monster/v1/series?_limit=10"
# head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

# guids = {"payload": {"title": "body_type"}}
# r1 = requests.post(url, headers=head, timeout=500000)

# print(r1.text)
