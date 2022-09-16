import requests
from requests import HTTPError


def download_csv(file, url):
    with open(file, 'wb') as f:
        resp = requests.get(url + file)
        try:
            resp.raise_for_status()
        except HTTPError as e:
            return False
        f.write(resp.content)
    return True
