import aiohttp
import asyncio
import json
import os

# group = "tits_size"
# guids = {
#     "guids": [
#     "af076d1f-0286-4db8-9dd9-d94809e89005", 
#     "4f837bde-d89c-4987-8ac0-feacda5ba1c7"
#   ]
# }
bearer = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2ODk4NjEyNTksIklkZW50aXR5Ijp7IkdVSUQiOiIzYTZjYmRiMy0wYmFiLTQxNDQtOWViNC00NDc0NWExY2ZiYmIiLCJMb2dpbiI6ImFkbWluIiwiUm9sZXMiOlt7IkdVSUQiOiJmMmQyNDA4Zi03MDc1LTRmOTctOTVlZi0zMjgzMWJiMDgyZjQiLCJOYW1lIjoiYWRtaW4iLCJUaXRsZSI6ImFkbWluIiwiRGVzY3JpcHRpb24iOiIifV19fQ.LIgtSc_NrWHvaEsoBUVQMPvFMokTSsrFsPSqPOEVE_w'


def get_head(bearer):
    head = {"Authorization": f"bearer {bearer}", "Content-Type": "application/json"}
    return head


async def fetch_meta(session, url, headers, guid, group, meta_dir):
    json_data = json.dumps({"guids": [guid]})
    
    async with session.post(url, headers=headers, data=json_data, timeout=500000) as response:
        if response.status == 200:
            json_data = await response.json()
            folder_path = f"{meta_dir}/{group}/{guid}"
            os.makedirs(folder_path, exist_ok=True)
    
            with open(os.path.join(folder_path, "meta.json"), "w") as file:
                json.dump(json_data["data"], file, indent=2)
            print(f"Saved {guid}.json")
    
        else:
            print(response)
            print(f"Failed to fetch meta data for {guid}")


async def get_meta(bearer, guids, group, meta_dir):
    URL = 'https://yapics.collect.monster/v1/meta/picsets'
    head = get_head(bearer)
    async with aiohttp.ClientSession(trust_env = True) as session:
        tasks = [fetch_meta(session, URL, head, guid, group, meta_dir) for guid in guids["guids"]]
        await asyncio.gather(*tasks)


# asyncio.run(get_meta(bearer, guids, group))
