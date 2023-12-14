import json
import requests
    
    
headers = {"Content-Type": "application/json"}
data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}
url = "https://yapics2.dev.collect.monster/v1/login"
r = requests.post(url, data=json.dumps(data_log), headers=headers)
token = eval(r.text)["token"]

# # Create series
# head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
# data = {"payload": {"title": "hair_type"}}
# r = requests.put('https://yapics2.dev.collect.monster/v1/series', headers=head, data=json.dumps(data))
# print(r.text)

# # Create category
# head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
# data = {"payload": {"title": 'hair_type trash', "series": {"title": "hair_type"}}}
# r = requests.put('https://yapics2.dev.collect.monster/v1/category', headers=head, data=json.dumps(data))
# print(r.text)


# # Update category
# head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
# data = {"payload": {"title": "small tits", "series": {"title": "tits_size"}}}
# r = requests.patch('https://yapics2.dev.collect.monster/v1/category/d3cfeeeb-595d-451e-8d1f-4ef82c4f6f36', headers=head, data=json.dumps(data))
# print(r.text)

# # Get series
# head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
# r = requests.get('https://yapics2.dev.collect.monster/v1/series?_limit=10', headers=head)
# print(r.text)

# Get picsets
head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
data = {"guids": ["5152b9bc-135d-4fba-b725-3ee3edb18f65"]}
r = requests.post('https://yapics2.dev.collect.monster/v1/meta/picsets', headers=head, data=json.dumps(data))
with open('meta.json', 'w') as f:
    f.write(r.text)


