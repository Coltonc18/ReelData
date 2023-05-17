import concurrent.futures
import json
import os
import pickle
import re
import time
from queue import Queue
from threading import Thread

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Initialize the maximum amount of Threads to scrape with
# For PCs with lots of RAM and CPU power, this can be around 1000
MAX_THREADS = 1000

# Intitialize the Queue which will be used to write to a file in a Thread-Safe manner
csv_queue = Queue()

'''
main is for testing purposes only
DELETE BEFORE SUBMISSION AND CALL FROM main.py
'''
def main():
    scrape_top_tier_actors(pages=['alist'], test=False)

    pass


def scrape_top_tier_actors(pages, test=False, prefix='data/'):
    '''
    Scrapes A-List, Top 100, and Top 1000 actor lists from imdb.com and saves them as sets to .pickle files.
    Sets stored in a pickle are used for their O(1) efficiency when determining if an item is in the set.

        Parameters:
                pages (list): Pages which should be scraped. Options are alist, top_100, and top_1k.
                test (bool): Default False, whether the method should assure that all top 100 actors also appear in top 1k set. For correct usage, pages should contain both 'top_100' and 'top_1k' when True

        Returns:
                None
    '''
    # Scrape A-List Actors
    alist_actors = set()
    if 'alist' in pages:
        # Get the BeautifulSoup object from the page with A-List actors
        alist_url = 'https://www.imdb.com/list/ls044030121/'
        page_alist = requests.get(alist_url)
        soup_alist = BeautifulSoup(page_alist.content, 'html.parser')
        # Iterate through each list item on the page and scrape the actor's name
        for child in range(1, 14):
            alist_actors.add(soup_alist.select(f'div.lister-list > div:nth-child({child}) > div.lister-item-content > h3 > a')[0].get_text().strip())
        # Save to a pickle file for easy access later
        with open(f'{prefix}alist_actors.pickle', 'wb') as alist_pickle:
            print(f'dumping {alist_actors}')
            pickle.dump(alist_actors, alist_pickle)

    # Scrape Top-100 Actors
    actors_100 = set()
    if 'top_100' in pages:
        # Get the BeautifulSoup object from the page with the top 100 actors
        top_100_url = 'https://www.imdb.com/list/ls050274118/'
        page_100 = requests.get(top_100_url)
        soup_100 = BeautifulSoup(page_100.content, 'html.parser')
        # Iterate through each list item on the page and get the actor's name
        for child in range(1, 101):
            actors_100.add(soup_100.select(f'div.lister-list > div:nth-child({child}) > div.lister-item-content > h3 > a'
                                        )[0].get_text().strip())
        # Save to a pickle file for easy access later
        with open(f'{prefix}top_100_actors.pickle', 'wb') as top100_pickle:
            pickle.dump(actors_100, top100_pickle)

    # Scrape Top-1000 Actors
    actors_1k = set()
    if 'top_1k' in pages:
        # Go to each of the ten pages which hold the top 1000 actors
        for page in range(1, 11):
            # Get the page and BeautifulSoup object
            top_1k_url = f'https://www.imdb.com/list/ls058011111/?page={page}'
            page_1k = requests.get(top_1k_url)
            soup_1k = BeautifulSoup(page_1k.content, 'html.parser')
            # Iterate through each list item on the page and get the actor's names
            for child in range(1, 101):
                actors_1k.add(soup_1k.select(f'div.lister-list > div:nth-child({child}) > div.lister-item-content > h3 > a'
                                            )[0].get_text().strip())
        # Save to a pickle file for easy access later
        with open(f'{prefix}top_1k_actors.pickle', 'wb') as top1k_pickle:
            pickle.dump(actors_1k, top1k_pickle)
    
    # Test to ensure all actors in the 100 list also appear in the 1k list
    # TODO: #2 Move to tests.py
    if test and ('top_100' in pages) and ('top_1k' in pages):
        for actor in actors_100:
            if actor not in actors_1k:
                print(f'{actor} is not in the top 1000, but is in the top 100')
    elif test and (('top_100' not in pages) or ('top_1k' not in pages)):
        print('Could not check for page accuracy: top_100 and top_1k were not selected')


def consume_queue(filepath):
    '''
    Using a helper thread, empties the objects from ``csv_queue`` into the designated filepath.
    Will stop running when "done" is placed into the Queue.

        Parameters:
                filepath (str): Filepath to write to while emptying the Queue

        Returns:
                None: when "done" is passed into the Queue
    '''
    with open(filepath, 'a') as f:
        # Run until "done" is passed into our Queue
        while True:
            if not csv_queue.empty():
                item = csv_queue.get()
                if item == "done":
                    return
                else:
                    f.write(item)
            else:
                # If the queue is empty, sleep for a second to save resources before re-checking
                time.sleep(1)


def web_scraping_tomatoes(clear=True, verbose=True):
    '''
    Prepares the tomatoes_ratings.csv file to be written to by a Thread.
    Cleans the titles found in the movies_metadata.csv file and replaces spaces with underscores.
    Times and calls the subsequent methods to scrape rottentomatoes.com.
    Prints out helpful information as the program runs if verbose is True.
    
        Parameters:
                clear (bool): Default True, whether or not to clear the existing csv file and overwrite
                verbose (bool): Default True, whether to print out debug messages and progress updates while running

        Returns:
                None

        Helper Methods: 
            ``_download_data_tomatoes()``
            ``_access_page_tomatoes()``
            ``consume_queue()``
    '''
    # Set up csv file to store critics' ratings
    print('Setting up csv files...') if verbose else None
    if clear:
        with open('data/tomatoes_ratings.csv', 'w') as f:
            f.write('id,title,scraped_title,rating\n')
        print('Cleared CSV') if verbose else None
    
    # Import the titles from movies_metadata and replace spaces with underscores
    titles = pd.read_csv('data/movies_metadata.csv', usecols=['id', 'title'])
    titles['scraped_title'] = titles['title'].apply(lambda a: str(a).lower().replace(' ', '_'))

    # Initializes a thread to constantly scan csv_queue for lines to be added to the csv
    print('Starting writer thread...') if verbose else None
    Thread(target=consume_queue, args=('data/tomatoes_ratings.csv',)).start()
    
    # Starts a timer and sends the titles to be scraped
    print(f'Starting to scrape with up to {MAX_THREADS} threads...') if verbose else None
    t0 = time.time()
    _download_data_tomatoes(titles, verbose)
    t1 = time.time()

    # Print the time elapsed to scrape all pages
    print(f"    Took {t1-t0} seconds to scrape {len(titles)} pages.") if verbose else None


def _download_data_tomatoes(titles, verbose=True):
    '''
    Creates a thread pool using the ``ThreadPoolExecutor`` and maps each thread to run ``_access_page_tomatoes()``

        Parameters:
                titles (DataFrame): Contains columns "id" and "title", identifying which titles to scrape for
                verbose (bool): Default True, whether to print out debug messages and progress updates while running
        
        Returns:
                None
    '''
    # Don't start more threads than there are pages to scrape
    threads = min(MAX_THREADS, len(titles))

    # Initialize the ThreadPool to concurrently scrape many pages at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Map each thread to a row in titles though the iterrows() generator function
        executor.map(_access_page_tomatoes, titles.iterrows())

    # Close the csv_writer thread when all pages have been scraped
    csv_queue.put("done")


def _access_page_tomatoes(movie, verbose=True):
    '''
    Cleans the title and scrapes https://rottentomatoes.com/m/title (if it exists) for the critic rating of that movie.

        Parameters:
                movie (int, Series): The index of the row containing the title and id, and a Series of that data
                verbose (bool): Default True, whether to print out debug messages and progress updates while running

        Returns:
                None when a page cannot be accessed
    '''
    # Convert the row to a dictionary
    movie = movie[1].to_dict()
    # Get the title of the movie as it appears in the dataset
    title = movie['scraped_title']
    
    # Remove "the" and "a" if they are the first words of the title
    title = title[4:] if title[:3] == 'the' else title
    title = title[2:] if title[:1] == 'a' else title
    # Remove any special characters that will be omitted from the url
    title = re.sub(r"['\".:,-]", '', title)
    title = re.sub(r"__", '_', title)
    
    # Append the title to the base url
    # NOTE: Movies that have duplicate titles on rottentomatoes.com will not fit this url model because
    #       they have an identifying tag (7 digit number) as a prefix on the title, and therefore cannot 
    #       be scraped because we do not know how to get or calculate that tag
    url = 'https://rottentomatoes.com/m/' + title

    # Send a request to get the page
    page = requests.get(url=url)

    # If we have an error code from the site, return empty-handed
    if page.status_code != 200:
        # print(f'404: Page not Found ({title})') if page.status_code == 404 else f'Error {page.status_code}'
        return

    # Parse the html from the page using the BeautifulSoup library
    soup = BeautifulSoup(page.content, 'html.parser')

    # Because the expert rating text is inside of a script, we need to use CSS selectors to find that script
    # And then the json library to parse the script into a more readable python object
    data = json.loads(soup.select('script#score-details-json[type="application/json"]')[0].text)

    # Place the movie title, from the parsed json, into the CSV Writer Queue to be appended to the csv file
    csv_queue.put(f'{movie["id"]},{movie["title"]},{title},{data["scoreboard"]["tomatometerScore"]["value"]}\n')

''' The following methods containing the suffix "_imdb" aim to scrape imdb.com, which is not only impossible 
    because of the fragility of their servers, but also because the ratings on imdb.com are given by users, 
    not experts, and therefore do not provide any contrastable data for us to use in conjunction with the data 
    we had already collected. Therefore, the below is considered dead code. 
'''


def web_scraping_imdb():
    '''
    Deprecated
    ----------
    
    Sets up CSV files and begins the scraping process for imdb.com.

        Parameters:
                None

        Returns:
                None
    '''
    print('Setting up csv files...')
    # If the file exists, overwrite whatever is there with a new CSV column header
    if os.stat('data/imdb_ratings.csv').st_size == 0:
        with open('data/imdb_ratings.csv', 'w') as f:
            f.write('imdb_id,rating\n')
    
    # Gather the imdb_ids from movies_metadata and convert to a list
    imdb_ids = pd.read_csv('data/movies_metadata.csv.gz', usecols=['imdb_id'])['imdb_id'].tolist()

    print('Starting writer thread...')
    Thread(target=consume_queue, args=('data/imdb_ratings.csv',)).start()
    
    print('Starting to scrape...')
    t0 = time.time()
    ratings = download_data_imdb(imdb_ids)
    t1 = time.time()
    print(f"    Took {t1-t0} seconds to scrape {len(imdb_ids)} pages.")


def download_data_imdb(imdb_ids):
    '''
    Deprecated
    ----------

    Maps the ``ThreadPoolExecutor`` to access each page in ``imdb_ids``.

        Parameters:
                imdb_ids (list): Contains all ``imdb_ids`` to be scraped from imdb.com

        Returns:
                None
    '''
    # Don't start more threads than there are pages to scrape
    threads = min(MAX_THREADS, len(imdb_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(access_page_imdb, imdb_ids)

    # Put "done" in the csv_queue when done scraping to close the Thread
    csv_queue.put("done")


def access_page_imdb(id):
    '''
    Deprecated
    ----------

    Accesses a page on imdb.com, scrapes the rating from the page, and adds it to the csv file.

        Parameters:
                id (str): One imdb_id to be scraped

        Returns:
                None
    '''
    # The URL for any movie on imdb.com is:
    url = 'https://imdb.com/title/' + id

    # It seems as though session-id needs to be changed periodically to avoid certain errors
    # To get a new one, open a imdb.com page in a browser
        # Hit F12, go to Network tab and find first line in table
        # Scroll to find sessionid under Request Headers and paste below
    session_id='134-2074330-3006658'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34', 
               'set-cookie': f'session-id={session_id}; Domain=.imdb.com; Expires=Tue, 01 Jan 2036 08:00:01 GMT; Path=/'}
    
    # Send a request to the site to get the page content
    # Headers are used as artificial cookies to impersonate a real request by a browser
    page = requests.get(url=url, headers=headers)
    
    # If we cannot access the page, try again in 10 seconds
    # While scraping, usually this would occur when the imdb servers crashed as a result of too many requests
    # Eventually, they would reboot and we could continue scraping
    # This is probably unethical but...
    while page.status_code != 200:
        print(id, 'not accessible. Error code:', page.status_code)
        time.sleep(10)
        page = requests.get(url=url, headers=headers)

    # Convert the page content to a BeautifulSoup object
    soup = BeautifulSoup(page.content, 'html.parser')

    # Select the rating from the page and place it inot the csv_queue
    csv_queue.put(f'{id},{soup.select("span.sc-bde20123-1.iZlgcd")[0].get_text().strip()}\n')


def download_data_slow(imdb_ids):
    '''
    Deprecated
    ----------

    Very slow approach to scraping all pages. Extrapolated runtimes estimated at upwards of ``24 hours`` for all movies on a very capable PC.
    '''
    # Loop through each id in the list, and scrape the data from them. One. At. A. Time.
    for id in imdb_ids:
        access_page_imdb(id)
    csv_queue.put("done")

if __name__ == "__main__":
    main()
