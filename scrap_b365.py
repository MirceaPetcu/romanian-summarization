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
auth_source = "focuspress_news"  
database_name = "focuspress_news"
uri = f"mongodb://{username}:{password}@{host}:{port}/{database_name}?authSource={auth_source}"
client = pymongo.MongoClient(uri)
db = client[database_name]
collection = db["news"]  
ads = ['Promoveaza-te prin intermediul FocusPress si ai acces la milioane de vizitatori.','Promoveaza-te alaturi de FocusPress. Reclama ta poate fi aici.']
category = 'politica'
no_of_pages = 118


# from bson.objectid import ObjectId  

# document_id = ObjectId("65ea3d4edb5c38f0e751a80f") 

# delete_result = collection.delete_one({"_id": document_id})

# documents = collection.find()
# for doc in documents:
#     print("Document:", doc)


# proxies = {
# 	'http': 'http://yourhttpproxyserver.com:8080'
# }

def get_headers():
    headers = {
				'User-Agent': ua.random
				}
    return headers

def filter_link(link):
    if 'wp-content' in link:
        return False
    return True

try:
# de adaugat human-like headers alternat
	for page in range(118,no_of_pages+1):
		time.sleep(random.randint(10, 18))
		if page == 1:
			response = requests.get(f'https://b365.ro/post-sitemap.xml',headers=get_headers()) 
		else:
			response = requests.get(f'https://b365.ro/post-sitemap{page}.xml',headers=get_headers())

		if response.status_code != 200:
			print('Failed to fetch the page of news', page)
			print(f'Status code: {response.status_code}')
			continue

		# html_content = response.content.decode('utf-8')
		html_content = response.content
		soup = BeautifulSoup(markup=html_content, features='lxml-xml')
		links = [link.text for link in soup.find_all('loc')]

		filtered_links =  [link for link in links if filter_link(link)]

		if page == 118:
			filtered_links = filtered_links[232:]
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
			title_content = soup_link.find('h1', class_='font-titles')
			if title_content is None:
				continue
			# check if the news is not an ad
			if '(P)' in title_content.text:
				continue
			
			main_div = soup_link.find('div', class_='single__content font-main')
			first_h2 = main_div.find('h2')
			if first_h2 is None or first_h2 == []:
				continue
			try:
				paragraphs_befor_first_h2 = first_h2.find_all_previous('p')
			except Exception as e:
				continue
			# the sample have to contain only one paragraph before the first h2
			if len(paragraphs_befor_first_h2) != 1:
				continue

			# get summary
			summary = paragraphs_befor_first_h2[0].text
	
			# get content
			paragraphs = main_div.find_all('p')
			content = ''
			for paragraph in paragraphs[1:]:
				content += paragraph.text
				content += ' '

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

			if not has_summary:
				continue
			
			
			sample = {
				'Category': '-',
				'Title': title_content.text,
				'Content': content,
				'Summary': summary,
				'href': link,
				'Source': 'b365.ro',
			}

			# Insert a single document
			inserted_result = collection.insert_one(sample)
			print(f"Inserted element with number {j} and page {page} and ID:", inserted_result.inserted_id )
   
except Exception as e:
    print(str(e))
finally:
	client.close()
	print(f'The script ended at page: {page} and news with number: {j}')
