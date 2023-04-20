import concurrent.futures
import json
import os
import re
import time
from queue import Queue
from threading import Thread

import pandas as pd
import requests
from bs4 import BeautifulSoup

MAX_THREADS = 1000
csv_queue = Queue()

'''
Main is for testing purposes only
DELETE BEFORE SUBMISSION AND CALL FROM main.py
'''
def main():
    # DO NOT RUN UNLESS U WANT 2 HRS OF COMPUTER LOCKUP
    # web_scraping_tomatoes()
    pass

'''
Using a thread, empties the objects in csv_queue into the designated filepath
Will stop running when "done" is placed into the Queue
'''
def consume_queue(filepath):
    with open(filepath, 'a') as f:
        while True:
            if not csv_queue.empty():
                item = csv_queue.get()
                if item == "done":
                    return
                else:
                    f.write(item)

'''
Prepares the tomatoes_ratings.csv file to be written to by a Thread
Cleans the titles found in the movies_metadata.csv file and replaces spaces with underscores
Times and calls the subsequent methods to scrape rottentomatoes.com
Helper Methods: 
    _download_data_tomatoes()
    _access_page_tomatoes()
'''
def web_scraping_tomatoes(clear=True):
    print('Setting up csv files...')
    if clear:
        with open('data/tomatoes_ratings.csv', 'w') as f:
            f.write('title,rating\n')
        print('Cleared CSV')
    
    titles = pd.read_csv('data/movies_metadata.csv.gz', usecols=['original_title'])['original_title']
    titles = titles.apply(lambda a: str(a).lower().replace(' ', '_')).tolist()

    print('Starting writer thread...')
    Thread(target=consume_queue, args=('data/tomatoes_ratings.csv',)).start()
    
    print(f'Starting to scrape with up to {MAX_THREADS} threads...')
    t0 = time.time()
    _download_data_tomatoes(titles)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(titles)} pages.")

'''
Takes a list of titles from download_data_tomatoes()

Creates a thread pool using the ThreadPoolExecutor and maps each thread to run _access_page_tomatoes()
with each title in the list provided as the parameters to iterate through.
'''
def _download_data_tomatoes(titles):
    threads = min(MAX_THREADS, len(titles))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(_access_page_tomatoes, titles)
    # Close the csv_writer thread when all pages have been scraped
    csv_queue.put("done")

'''
Takes a single title, cleans it from special characters and preceeding "the"s or "a"s, and scrapes
https://rottentomatoes.com/m/title (if it exists) for the critic rating of that movie.

Does an empty return if the page is not accessible and prints out the page's error message.
'''
def _access_page_tomatoes(title): 
    title = re.sub(r"['\".:,-]", '', title)
    title = title[4:] if title[:3] == 'the' else title
    title = title[2:] if title[:1] == 'a' else title

    url = 'https://rottentomatoes.com/m/' + title

    page = requests.get(url=url)
    if page.status_code != 200:
        print(f'404: Page not Found ({title})') if page.status_code == 404 else 'Error (not 404)'
        return

    soup = BeautifulSoup(page.content, 'html.parser')
    
    data = json.loads(soup.select('script#score-details-json[type="application/json"]')[0].text)

    csv_queue.put(f'{title},{data["scoreboard"]["tomatometerScore"]["value"]}\n')


'''
The following methods containing the suffix "_imdb" will be deleted soon.
They aim to scrape imdb.com, which is not only impossible because of the fragility of their servers,
but also because the ratings on imdb.com are given by users, not experts, and therefore do not
provide any contrastable data for us to use in conjunction with the data we have already collected.

These methods represent upwards of 12 hours of hard work and will be mourned greatly.
May they forever rest in peace.
'''
def web_scraping_imdb():
    print('Setting up csv files...')
    if os.stat('data/imdb_ratings.csv').st_size == 0:
        with open('data/imdb_ratings.csv', 'w') as f:
            f.write('imdb_id,rating\n')
    
    imdb_ids = pd.read_csv('data/movies_metadata.csv.gz', usecols=['imdb_id'])['imdb_id'].tolist()

    print('Starting writer thread...')
    Thread(target=consume_queue, args=('data/imdb_ratings.csv',)).start()
    # writer = Thread(target=consume_queue())
    # writer.setDaemon(True)
    # writer.start()
    
    print('Starting to scrape...')
    t0 = time.time()
    ratings = download_data_imdb(imdb_ids)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    # print('Starting to scrape slowly...')
    # t0 = time.time()
    # ratings = download_data_slow(imdb_ids)
    # t1 = time.time()
    # print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")

    return ratings

def access_page_imdb(id):
    url = 'https://imdb.com/title/' + id
    # Seems as though session-id needs to be changed periodically to avoid certain errors
    # To get a new one, open a imdb.com page in a browser
        # Hit F12, go to Network tab and find first line in table
        # Scroll to find sessionid under Request Headers and paste below
    session_id='134-2074330-3006658'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34', 
               'set-cookie': f'session-id={session_id}; Domain=.imdb.com; Expires=Tue, 01 Jan 2036 08:00:01 GMT; Path=/'}
    page = requests.get(url=url, headers=headers)
    while page.status_code != 200:
        print(id, 'not accessible. Error code:', page.status_code)
        time.sleep(10)
        page = requests.get(url=url, headers=headers)

    soup = BeautifulSoup(page.content, 'html.parser')
    # If more data was wanted from this imdb page, this is where it should be scraped
    csv_queue.put(f'{id},{soup.select("span.sc-bde20123-1.iZlgcd")[0].get_text().strip()}\n')

def download_data_imdb(imdb_ids):
    threads = min(MAX_THREADS, len(imdb_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(access_page_imdb, imdb_ids)
    csv_queue.put("done")

# Used for comparison to multithreaded solution
def download_data_slow(imdb_ids):
    for id in imdb_ids:
        access_page_imdb(id)
    csv_queue.put("done")

if __name__ == "__main__":
    main()
