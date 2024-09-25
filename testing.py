import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import re  # To handle new line replacements

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Define the file containing URLs and the output file
input_file = 'cleaned_normalized_urls.txt'
output_file = 'scraped_data_with_metadata.txt'

# Read URLs from input file
with open(input_file, 'r') as f:
    urls = [line.strip() for line in f.readlines()]

# Function to load and scrape web pages with LangChain's WebBaseLoader
def scrape_page_with_langchain(urls):
    loader = WebBaseLoader(
        web_paths=urls[:4],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(name=['h1', 'h2', 'h3', 'p', 'meta', 'title'])
        )
    )
    documents = loader.load()
    return documents

# Function to clean the content by removing extra new lines
def clean_text(content):
    # Replace multiple newlines with a single newline
    return re.sub(r'\n\s*\n+', '\n', content).strip()

# Function to save scraped data and metadata into a text file
def save_scraped_data(output_file, documents):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for doc in documents:
            # Write metadata
            metadata = doc.metadata
            outfile.write(f"URL: {metadata.get('source', 'Unknown')}\n")
            outfile.write(f"Title: {metadata.get('title', 'Unknown')}\n")
            
            # Clean and write the scraped content
            content = clean_text(doc.page_content)
            outfile.write(f"Content:\n{content}\n\n")
            outfile.write("="*80 + "\n\n")

# Scrape the web pages
scraped_documents = scrape_page_with_langchain(urls)

# Save the scraped data along with metadata, cleaning the content
save_scraped_data(output_file, scraped_documents)

print(f"Scraping completed. Cleaned data and metadata saved to {output_file}")
