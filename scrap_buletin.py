import requests 
from bs4 import BeautifulSoup 
import random
import time
import pymongo
from fake_useragent import UserAgent


ua = UserAgent()
host = "localhost"
port = 27017
username = "mircea"
password = "licenta"  
auth_source = "buletin_db"  
database_name = "bultetin_db"
uri = f"mongodb://{username}:{password}@{host}:{port}/{database_name}" 
client = pymongo.MongoClient(uri)
db = client[database_name]
collection = db["news_buletin"]  
no_of_pages = 8



def get_headers():
    headers = {
				'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0',
				}
    return headers

def filter_link(link):
    if 'toate-articolele' in link:
        return False
    if 'wp-content' in link:
        return False
    return True

try:
# de adaugat human-like headers alternat
	for page in range(7,1,-1):
		time.sleep(random.randint(10, 18))
		if page == 1:
			response = requests.get(f'https://buletin.de/bucuresti/post-sitemap.xml',headers=get_headers()) 
		else:
			response = requests.get(f'https://buletin.de/bucuresti/post-sitemap{page}.xml',headers=get_headers())

		if response.status_code != 200:
			print('Failed to fetch the page of news', page)
			print(f'Status code: {response.status_code}')
			continue

		# html_content = response.content.decode('utf-8')
		html_content = response.content
		soup = BeautifulSoup(markup=html_content, features='lxml-xml')
		links = [link.text for link in soup.find_all('loc')]

		filtered_links =  [link for link in links if filter_link(link)]
		if page == 7:
			filtered_links = filtered_links[124:]
		print(len(filtered_links))
		# get content for each news link
		for j,link in enumerate(filtered_links):
		
			time.sleep(random.randint(8, 16))
			link_response = requests.get(link,headers=get_headers())
	
			if link_response.status_code != 200:
				print('Failed to fetch the page of the single news', link)
				print(f'Status code: {link_response.status_code}')
				continue

			soup_link = BeautifulSoup(link_response.content, 'html.parser')
			# print(link)
	
			# check if the news is an ad
			powered_by = soup_link.find('div', class_='powered_by')
			if powered_by is not None:
				continue
			
			# get title
			title_content = soup_link.find('h1', class_='jl_head_title')

			if title_content is None:
				continue
			# check if the news is not an ad
			if '(P)' in title_content.text:
				continue
			
			

			main_div = soup_link.find('div', class_='post_content jl_content')
			if main_div is None:
				continue

			# get summary
			paragraphs = main_div.find_all('p')
			if paragraphs is None or len(paragraphs) == 0:		
				continue

			if "ERATĂ//" in paragraphs[0].text:
				paragraphs = paragraphs[1:]

			summary = paragraphs[0].find('strong')
			if summary is None:
				continue
			if len(summary.text.split()) < 15:
				continue
 
			summary = summary.text
   
			# get content
			content = ''
			for i,paragraph in enumerate(paragraphs[1:]):
				if i == len(paragraphs[1:]):
					if 'CITEȘTE ȘI' in paragraph.text:
						break
				content += paragraph.text
				content += ' '
			content = content.replace('\xa0', ' ')
   
			# check if the news has a summary
			has_summary = True
			# print(title_content.text)
			# print(summary)
			# print(content)
			length_content = len(content.split())
			length_summary = len(summary.split())
			# print(length_content)
			# print(length_summary)
			# print(length_content/length_summary)	
			if length_content/length_summary < 2.0:
				has_summary = False
			
			if length_summary < 15: 
				has_summary = False
			if length_summary >= 20:
				has_summary = False
    
			if not has_summary:
				continue

			print(length_summary)
   
			sample = {
				'Category': '-',
				'Title': title_content.text,
				'Content': content,
				'Summary': summary,
				'href': link,
				'Source': 'buletin.de/bucuresti.ro',
			}

			# Insert a single document
			inserted_result = collection.insert_one(sample)
			print(f"Inserted element with number {j} and page {page} and ID:", inserted_result.inserted_id )

except Exception as e:
    print(str(e))
finally:
	client.close()
	print(f'The script ended at page: {page} and news with number: {j}')
