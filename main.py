import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
import os
from threading import Thread
import _thread
from queue import Queue
import random

MAX_THREADS = 1024
csv_queue = Queue()

def main():
    # DO NOT UNCOMMENT THE FOLLOWING LINE WITHOUT CAUTION
    # compress_csv()

    web_scraping()
    # merge_data()

def merge_data():
    # Most datasets have id as the common denominator to merge on
    # ratings.csv has UserId and MovieId, MovieId is the same as id, but will need to be groupby'ed before merging to find avg rating for each movie
    # Metadata.csv contains the correct imdbId for webscraping as well as the usual id
    credits_df = pd.read_csv('data/credits.csv.gz', compression='gzip')
    links_df = pd.read_csv('data/links.csv.gz', compression='gzip')
    rating_df = pd.read_csv('data/ratings.csv.gz', compression='gzip')
    metadata_df = pd.read_csv('data/movies_metadata.csv.gz', compression='gzip')

    rating_avg_df = rating_df.groupby('movieId')['rating'].mean().reset_index()
    
    master_df = pd.merge(credits_df, metadata_df, on='id')

    master_df = pd.merge(master_df, rating_avg_df, left_on='id', right_on='movieId')

    master_df = pd.merge(master_df, links_df, on='movieId')

    # Drop unnecessary columns
    master_df.drop(['id', 'movieId'], axis=1, inplace=True)
    
    # Save merged dataframe to CSV
    master_df.to_csv("master_dataset.csv", index=False)

def compress_csv():
    for filename in os.listdir('data'):
        filename = os.path.join('data', filename)
        df = pd.read_csv(filename, low_memory=False)
        df.to_csv(f'{filename}.gz', compression='gzip')

def consume_queue():
    with open('data/imdb_ratings.csv', 'a') as f:
        while True:
            if not csv_queue.empty():
                item = csv_queue.get()
                if item == "done":
                    return
                else:
                    f.write(item)         

def web_scraping():
    print('Entering Scraping')
    with open('data/imdb_ratings.csv', 'w') as f:
        f.write('imdb_id,rating\n')
    
    print('Parsing csv')
    imdb_ids = pd.read_csv('data/movies_metadata.csv.gz', compression='gzip', usecols=['imdb_id'])['imdb_id'].tolist()

    print('Initializing writer')
    # writer = Thread(target=consume_queue())
    _thread.start_new_thread(consume_queue, ())
    # print('Making Daemonic')
    # writer.setDaemon(True)
    # print('Starting Writer')
    # writer.start()
    
    print('Starting to scrape quickly...')
    t0 = time.time()
    ratings = download_data(imdb_ids)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    # print('Starting to scrape slowly...')
    # t0 = time.time()
    # ratings = download_data_slow(imdb_ids)
    # t1 = time.time()
    # print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    return ratings

def access_page(id):
    url = 'https://imdb.com/title/' + id
    # Seems as though session-id needs to be changed periodically to avoid certain errors
    # To get a new one, open a imdb.com page in a browser
        # Hit F12, go to Network tab and find first line in table
        # Scroll to find sessionid under Request Headers and paste below
    session_id='134-2074330-3006658'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34', 
               'set-cookie': f'session-id={session_id}; Domain=.imdb.com; Expires=Tue, 01 Jan 2036 08:00:01 GMT; Path=/'}
    page = requests.get(url=url, headers=headers)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        # If more data was wanted from this page, this is where it should be scraped
        # with open('data/imdb_ratings.csv', 'a') as f:
        #     f.write(f'{id},{soup.select("span.sc-bde20123-1.iZlgcd")[0].get_text().strip()}\n')
        csv_queue.put(f'{id},{soup.select("span.sc-bde20123-1.iZlgcd")[0].get_text().strip()}\n')
    elif page.status_code == 503:
        print('Error 503: Service Unavailible. Please try again later')
        exit(503)
    else:
        print(id, 'not accessible. Error code:', page.status_code)

def download_data(imdb_ids):
    threads = min(MAX_THREADS, len(imdb_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(access_page, imdb_ids)
    csv_queue.put("done")

def download_data_slow(imdb_ids):
    for id in imdb_ids:
        access_page(id)
    csv_queue.put("done")


if __name__ == "__main__":
    main()
