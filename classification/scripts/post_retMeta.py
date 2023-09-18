import requests
import json
from glob import glob

# stand = 'dev.'
stand = ""

# GROUPS = ['body_type', 'sex_positions', 'tits_size']
GROUPS = ["test"]
ROOT = "/home/timssh/ML/TAGGING/DATA/meta"

headers = {"Content-Type": "application/json"}
data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

if __name__ == "__main__":
    url = f"https://yapics.{stand}collect.monster/v1/login"
    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]

    def get_meta(picset):
        with open(picset, "r", encoding="utf-8") as js_f:
            my_js = json.load(js_f)
        return my_js

    for GROUP in GROUPS:
        paths = glob(ROOT + f"/{GROUP}/*/ret_meta.json")
        for js_path in paths:
            print(js_path)
            data = get_meta(js_path)
            # for item in data['items']:
            #     item['trained'].append({
            #             "group": "group_of_girls",
            #             "category": ['one girl'

            #             ]
            #         })
            url = f"https://yapics.{stand}collect.monster/v1/picset/trained"
            head = {"Authorization": f"token {token}"}
            r1 = requests.post(url, data=json.dumps(data), headers=head, timeout=500000)
            print(r1, r1.text)
            print("set checking")
            r1 = requests.post(
                f"https://yapics.{stand}collect.monster/v1/picset/checking",
                data=json.dumps({"guids": [js_path.split("/")[-2]]}),
                headers=head,
                timeout=500000,
            )
