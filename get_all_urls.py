import json
# from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
# linklist = ["https://iciec.isdb.org"]

# for link in linklist:
#     print(link)
#     loader_multiple_pages = WebBaseLoader(
#         web_paths=[link],
#         bs_kwargs=dict(
#             parse_only=bs4.SoupStrainer(
#                 name=['h1']
#             )
#         ))
#     docs = loader_multiple_pages.load()
#     print(docs)

import requests
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import urljoin

linklist = ["https://iciec.isdb.org"]
visited_links = set(linklist)  # Initialize visited links with the existing links

# Set up headers with a User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

for link in linklist:
    print(f"Visiting: {link}")
    
    # Fetch the webpage content with headers and disable SSL verification
    response = requests.get(link, headers=headers, verify=False)
    print(f"Response Code for {link}: {response.status_code}")
    
    if response.status_code == 200:  # Proceed only if the request was successful
        # Parse only the links
        for a in BeautifulSoup(response.content, 'html.parser', parse_only=SoupStrainer('a')):
            href = a.get('href')
            if href:
                full_url = urljoin(link, href)  # Create a full URL
                # print(f"Found href: {full_url}")
                
                # Check conditions for adding to the linklist
                if full_url not in visited_links and not full_url.endswith('.pdf') and full_url.startswith(('http://iciec.isdb.org','https://iciec.isdb.org')):
                    # if full_url.startswith(('http://', 'https://')):  # Check for valid URLs
                    visited_links.add(full_url)  # Mark as visited
                    linklist.append(full_url)  # Append the new link
                        # print(f"Added link: {full_url}")

print("Final linklist:")
print(linklist) exlude url shoud be add morning
