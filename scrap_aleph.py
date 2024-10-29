import requests 
from bs4 import BeautifulSoup 
import random
import time
import pymongo
from fake_useragent import UserAgent
from fp.fp import FreeProxy
import json


ua = UserAgent()
host = "localhost"
port = 27017
username = "mircea"
password = "licenta"  
database_name = "aleph_db"
uri = f"mongodb://{username}:{password}@{host}:{port}/{database_name}"
client = pymongo.MongoClient(uri)
db = client[database_name]
collection = db["news"]  
no_of_pages = 275
f = open('double_check_link.txt','a')

def get_headers():
	headers = {
				'User-Agent': ua.random
				}
	return headers

def filter_link(link):
	if 'wp-content' in link:
		return False
	if link.endswith('.pdf') or link.endswith('.jpg') or link.endswith('.png') or \
		link.endswith('.jpeg') or link.endswith('webp') or link.endswith('.gif'):
		return False
	return True

try:
	for page in range(244,no_of_pages+1):
		time.sleep(random.randint(10, 18))
		if page == 1:
			response = requests.get(f'https://alephnews.ro/post-sitemap.xml',headers=get_headers()) 
		else:
			response = requests.get(f'https://alephnews.ro/post-sitemap{page}.xml',headers=get_headers())

		if response.status_code != 200:
			print('Failed to fetch the page of news', page)
			print(f'Status code: {response.status_code}')
			continue

		html_content = response.content
		soup = BeautifulSoup(markup=html_content, features='lxml-xml')
		links = [link.text for link in soup.find_all('loc')]

		filtered_links =  [link for link in links if filter_link(link)]
		if page == 244:
			filtered_links = filtered_links[29:]
			print(f'Page {page} has {len(filtered_links)} news')
		print(len(filtered_links))
  
		# get content for each news link
		for j,link in enumerate(filtered_links):
			time.sleep(random.randint(10, 17))
			link_response = requests.get(link,headers=get_headers())
	
			if link_response.status_code != 200:
				print('Failed to fetch the page of the single news', link)
				print(f'Status code: {link_response.status_code}')
				f.write(link)
				f.write('\n')
				continue

			soup_link = BeautifulSoup(link_response.content, 'html.parser')
			# print(link)
	
			target_scripts = soup_link.find_all("script", {"type": "application/ld+json"})
			if target_scripts is None or len(target_scripts) < 4:
				f.write(link)
				f.write('\n')
				continue

			the_script = target_scripts[3].text
			if the_script is None:
				f.write(link)
				f.write('\n')
				continue

			the_script = the_script.replace('\n','')
   
			try:
				json_data = json.loads(the_script)
			except Exception as e:
				print(str(e))
				f.write(link)
				f.write('\n')
				continue

			# get title
			title_content = json_data['headline']

			if title_content is None:
				f.write(link)
				f.write('\n')
				continue
			# check if the news is not an ad
			if '(P)' in title_content:
				f.write(link)
				f.write('\n')
				continue

			main_div = soup_link.find('div', class_="post-content")
			if main_div is None:
				f.write(link)
				f.write('\n')
				continue


			# get summary
			ul = main_div.find('ul')
			if ul is None:
				f.write(link)
				f.write('\n')
				continue

			lis = ul.find_all('li')
			if lis is None:
				f.write(link)
				f.write('\n')
				continue
			if len(lis) == 0:
				f.write(link)
				f.write('\n')
				continue

			summary = ''
			for li in lis:
				summary += li.text
				summary += ' '
			summary = summary.replace('\n','')
			summary = summary.strip()


			# get content
			summary_plus_content = json_data['articleBody']
			summary_plus_content = summary_plus_content.strip()
			start_content = summary_plus_content.find(summary)+len(summary)-1
			end_content = summary_plus_content.find('Citește și')
			content = summary_plus_content[start_content:end_content]

			
			has_summary = True
			length_content = len(content.split())
			length_summary = len(summary.split())
			
			if length_content/length_summary < 2.0:
				has_summary = False

			if not has_summary:
				f.write(link)
				f.write('\n')
				continue
			
			if length_summary < 15:
				f.write(link)
				f.write('\n')
				continue

			# print(j)
			# print('TITLE:\n')
			# print(title_content)
			# print('SUMMARY:\n')
			# print(summary)
			# print('CONTENT:\n')
			# print(content)
			# print(length_content)
			# print(length_summary)
			# print(length_content/length_summary)	
			# print()
	
			sample = {
				'Category': link.split('/')[3],
				'Title': title_content,
				'Content': content,
				'Summary': summary,
				'href': link,
				'Source': 'alephnews.ro',
			}

			# Insert a single document
			inserted_result = collection.insert_one(sample)
			print(f"Inserted element with number {j} and page {page} and ID:", inserted_result.inserted_id )

except Exception as e:
	print(str(e))
finally:
	client.close()
	f.close()
	print(f'The script ended at page: {page} and news with number: {j}')
