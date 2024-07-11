from .config import download_url
import py7zr
import urllib.request

urllib.request.urlretrieve(download_url, "database.7z")

with py7zr.SevenZipFile('database.7z', mode='r') as z:
    z.extractall()