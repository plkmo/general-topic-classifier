import datetime
import random
import time
import csv
from bs4 import BeautifulSoup
from selenium import webdriver

def crawl_eastasiaforum(selective=True):
    driver = webdriver.Chrome()
    qualifying_tokens = ['singapore', 'singaporean']
    url_list = []
    page = 1
    while True:
        try:
            driver.get("https://www.eastasiaforum.org/category/topics/politics/page/%s/" % page)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            time.sleep(random.uniform(1, 2))
            posts = soup.findAll('div', class_='post front-post')

            if len(posts) == 0:
                break

            for post in posts:
                title = post.findAll('span', class_='info')[0]
                title_text = title.h2.text
                url = title.h2.a['href']
                date = title.time.text
                raw_abstract = post.findAll('section', class_='content')[0].text

                if selective:
                    for token in qualifying_tokens:
                        if (token in title_text.lower()) or (token in raw_abstract.lower()):
                            url_list.append(url)
                else:
                    url_list.append(url)
                    
        except Exception as e:
            print(e)
            break
        
        page += 1

    print("Total urls: %d" % len(url_list))
    data = []
    for url in url_list:
        print("Crawling %s..." % url)
        try:
            driver.get(url)
            time.sleep(random.uniform(1, 3))
            soup = BeautifulSoup(driver.page_source, "html.parser")
            article = soup.findAll('article', class_='full-post post')[0]
            title = article.header.h1.text
            date_ = article.header.time.text
            article_text_ = article.findAll('section', class_='content')[0].findAll('p')
            article_text = ""
            for p in article_text_[1:]:
                article_text += ' ' + p.text
            data.append([title, article_text])
        except Exception as e:
            print(e)
    
    try:
        driver.close()
    except:
        pass
    
    with open('eastasiaforum.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['title', 'text'])
        csv_writer.writerows([[title, text] for title, text in data])
    return

def crawl_theindependent():
    # kanna block ip
    driver = webdriver.Chrome()
    page = 1
    url_list = []
    while True:
        try:
            driver.get("https://theindependent.sg/news/singapore-news/singapolitics/page/%d/" % page)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            time.sleep(random.uniform(5, 7))
            articles = soup.findAll('div', class_='td_module_11 td_module_wrap td-animation-stack td-meta-info-hide')
            
            if len(articles) == 0:
                break

            for article in articles:
                url = article.findAll('h3', class_="entry-title td-module-title")[0]
                url_list.append(url.a['href'])

        except Exception as e:
            print(e)
            break

        page += 1

    print("Total urls: %d" % len(url_list))
    return

def crawl_asiaone():
    driver = webdriver.Chrome()
    driver.get('https://www.asiaone.com/tags/singapore-politics')
    error_count = 0

    # Get scroll height
    SCROLL_PAUSE_TIME = 6.5
    START_TIME = time.time()
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        load_more_button = driver.find_elements_by_xpath("//button[@class='ant-btn ant-btn-primary ant-btn-round']")
        try:
            load_more_button[0].click()
        except Exception as e:
            print(e)
            pass

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("REACHED SCROLL LIMIT.")
            break
        last_height = new_height

        if (time.time() - START_TIME) > 607: # hard stop after ~10 mins
            break

    # start extracting
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = soup.findAll('div', class_='ant-col ant-col-xs-24 ant-col-md-24 ant-col-lg-24')
    print("No. of articles found: %d" % len(articles))
    url_list = []

    for article in articles:
        url = article.findAll('li', class_='ant-list-item')[0].a['href']
        url_list.append("https://www.asiaone.com" + url)

    print("No. of urls: %d" % len(url_list))
    data = []
    for url in url_list:
        print("Getting... %s" % url)
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            title = soup.findAll('h1', class_='title')[0].text
            body = soup.findAll('div', class_='body')[0].text
            data.append([title, body])
        except Exception as e:
            print(e)
            pass

    driver.close()
    print("Saving...")
    with open('asiaone.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['title', 'text'])
        csv_writer.writerows([[title, text] for title, text in data])
    return

def crawl_coconuts():
    driver = webdriver.Chrome()
    driver.get('https://coconuts.co/singapore/news/politics/')
    error_count = 0

    # Get scroll height
    SCROLL_PAUSE_TIME = 6.5
    START_TIME = time.time()
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        try:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            load_more_button = driver.find_elements_by_class_name("button-loadmore")
            try:
                load_more_button[0].click()
                print("CLICKED LOAD MORE.")
                time.sleep(3)
            except Exception as e:
                print("ERROR CLICKING: ", e)
                pass

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("REACHED SCROLL LIMIT.")
                break
            last_height = new_height

            if (time.time() - START_TIME) > 1907: # hard stop after ~30 mins
                break
        except Exception as e:
            print("ERROR IN WHILE LOOP: ", e)


    # start extracting
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = soup.findAll('article', class_='coco-article row')
    print("No. of articles found: %d" % len(articles))
    url_list = []

    for article in articles:
        url = article.findAll('h5', class_='coco-article-title')[0].a['href']
        url_list.append(url)

    print("No. of urls: %d" % len(url_list))
    data = []
    for url in url_list:
        print("Getting... %s" % url)
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            title = soup.findAll('h1', class_='post-title')[0].text
            body = soup.findAll('div', class_='coco_post-content')[0].text
            data.append([title, body])
        except Exception as e:
            print(e)
            pass
    
    try:
        driver.close()
    except:
        pass

    print("Saving...")
    with open('coconuts.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['title', 'text'])
        csv_writer.writerows([[title, text] for title, text in data])
    return

def crawl_aljazeera():
    driver = webdriver.Chrome()
    driver.get('https://www.aljazeera.com/tag/politics/')
    error_count = 0

    # Get scroll height
    SCROLL_PAUSE_TIME = 6.5
    START_TIME = time.time()
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        try:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            load_more_button = driver.find_elements_by_xpath("//button[@class='show-more-button big-margin']")
            try:
                load_more_button[0].click()
                print("CLICKED LOAD MORE.")
                time.sleep(3)
            except Exception as e:
                print("ERROR CLICKING: ", e)
                pass

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("REACHED SCROLL LIMIT.")
                break
            last_height = new_height

            if (time.time() - START_TIME) > 1907: # hard stop after ~30 mins
                break
        except Exception as e:
            print("ERROR IN WHILE LOOP: ", e)


    # start extracting
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = soup.findAll('article', class_='gc u-clickable-card gc--type-post gc--list gc--with-image')
    print("No. of articles found: %d" % len(articles))
    url_list = []

    for article in articles:
        url = article.findAll('h3', class_='gc__title')[0].a['href']
        url_list.append("https://www.aljazeera.com" + url)

    print("No. of urls: %d" % len(url_list))
    data = []
    for url in url_list:
        print("Getting... %s" % url)
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            title = soup.findAll('h1')[0].text
            sub_title = soup.findAll('p', class_='article__subhead css-1wt8oh6')[0].text
            body = soup.findAll('div', class_='wysiwyg wysiwyg--all-content css-1vsenwb')[0].text
            data.append([title, sub_title.strip().strip('\n') + " " + body.strip().strip('\n')])
            #print(title, sub_title, body)
        except Exception as e:
            print(e)
            pass
    
    try:
        driver.close()
    except:
        pass

    print("Saving...")
    with open('aljazeera.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['title', 'text'])
        csv_writer.writerows([[title, text] for title, text in data])
    return

def crawl_worldpolitics():
    driver = webdriver.Chrome()
    crawl_url = 'https://www.worldpoliticsreview.com/search?query=singapore&order=desc&sort=score&page=%d'
    driver.get()
    page = 1
    START_TIME = time.time()
    while True:
        try:
            

            if (time.time() - START_TIME) > 1207: # hard stop after ~10 mins
                break
        except Exception as e:
            print("ERROR IN WHILE LOOP: ", e)


    # start extracting
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = soup.findAll('article', class_='gc u-clickable-card gc--type-post gc--list gc--with-image')
    print("No. of articles found: %d" % len(articles))
    url_list = []

    for article in articles:
        url = article.findAll('h3', class_='gc__title')[0].a['href']
        url_list.append("https://www.aljazeera.com" + url)

    print("No. of urls: %d" % len(url_list))
    data = []
    for url in url_list:
        print("Getting... %s" % url)
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            title = soup.findAll('h1')[0].text
            sub_title = soup.findAll('p', class_='article__subhead css-1wt8oh6')[0].text
            body = soup.findAll('div', class_='wysiwyg wysiwyg--all-content css-1vsenwb')[0].text
            data.append([title, sub_title.strip().strip('\n') + " " + body.strip().strip('\n')])
            #print(title, sub_title, body)
        except Exception as e:
            print(e)
            pass
    
    try:
        driver.close()
    except:
        pass

    print("Saving...")
    with open('aljazeera.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['title', 'text'])
        csv_writer.writerows([[title, text] for title, text in data])
    return

if __name__ == '__main__':
    crawl_coconuts()
    #crawl_theindependent()
    #crawl_asiaone()
    #crawl_aljazeera()