import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
import pickle
import os

MAX_THREADS = 100

def main():
    compress_csv()
    # web_scraping()

def merge_data():
    # Most datasets have id as the common denominator to merge on
    # ratings.csv has UserId and MovieId, MovieId is the same as id, but will need to be groupby'ed before merging to find avg rating for each movie
    # Metadata.csv contains the correct imdbId for webscraping as well as the usual id
    pass

def compress_csv():
    for filename in os.listdir('data'):
        filename = os.path.join('data', filename)
        df = pd.read_csv(filename, low_memory=False)
        df.to_csv(f'{filename}.gz', compression='gzip')

def web_scraping():
    imdb_ids = pd.read_csv('data/movies_metadata.csv.gz', compression='gzip', usecols=['imdb_id'])['imdb_id'].tolist()[:10]
    
    print('Starting to scrape quickly...')
    t0 = time.time()
    ratings = download_data(imdb_ids)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    print('Starting to scrape slowly...')
    t0 = time.time()
    ratings = download_data_slow(imdb_ids)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    return ratings

def access_page(id):
    url = 'https://imdb.com/title/' + id
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34'}
    page = requests.get(url=url, headers=headers)

    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        # If more data was wanted from the page, this is where it should be scraped
        with open('data/imdb_ratings.csv', 'a') as f:
            f.write(f'{id},{soup.select("span.sc-bde20123-1.iZlgcd")[0].get_text().strip()}\n')
    else:
        print(id, 'not accessible. Error code:', page.status_code)

def download_data(imdb_ids):
    with open('data/imdb_ratings.csv', 'w') as f:
        f.write('imdb_id,rating\n')

    threads = min(MAX_THREADS, len(imdb_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(access_page, imdb_ids)

def download_data_slow(imdb_ids):
    with open('data/imdb_ratings.csv', 'w') as f:
        f.write('imdb_id,rating\n')

    for id in imdb_ids:
        access_page(id)


if __name__ == "__main__":
    main()
