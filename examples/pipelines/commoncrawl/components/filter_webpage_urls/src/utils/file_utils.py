import os
import requests
import tarfile

from typing import List


def download_tar_file(blacklist_url: str, output_folder: str) -> None:
    """Download the blacklisted urls from the given url and extract them to the given folder."""
    response = requests.get(blacklist_url)
    response.raise_for_status()

    temp_file_path = "temp.tar.gz"
    with open(temp_file_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(temp_file_path, "r:gz") as tar:
        tar.extractall(output_folder)

    os.remove(temp_file_path)


def get_urls_from_file(blacklist_file: str) -> List[str]:
    try:
        with open(blacklist_file) as f:
            urls = f.readlines()
            return [url.strip() for url in urls]
    except FileNotFoundError:
        raise FileNotFoundError(f"Blacklist file {blacklist_file} does not exist")
