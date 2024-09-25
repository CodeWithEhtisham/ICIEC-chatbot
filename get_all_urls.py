import requests
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import urljoin
import urllib3

# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

linklist = ["https://iciec.isdb.org"]
visited_links = set(linklist)  # Initialize visited links with the existing links

# Set up headers with a User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

exclude_languages = ['/ar/', '/fr/']
# List to store excluded URLs (Arabic and French)
excluded_links = [
    "https://iciec.isdb.org/media-center",
    "https://iciec.isdb.org/knowledge-center/articles-of-agreement",
    "https://iciec.isdb.org/knowledge-center/annual-reports",
    "https://iciec.isdb.org/knowledge-center/annual-development-effectiveness-reports",
    "https://iciec.isdb.org/knowledge-center/rating-reports",
    "https://iciec.isdb.org/knowledge-center/brochures",
    "https://iciec.isdb.org/knowledge-center/reports",
    "https://www.isdbcareers.com/careers",
    "https://iciec.isdb.org/years",
    "https://iciec.isdb.org/all-products",
    "https://iciec.isdb.org/MRatingReports"
]

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
                if any(lang in full_url for lang in exclude_languages):
                    excluded_links.append(full_url)
                    print(f"Excluded non-English link: {full_url}")
                    continue  # Skip this URL
                # Check conditions for adding to the linklist (English URLs only)
                if full_url not in visited_links and not full_url.endswith(('.pdf','.docx','.xlsx','.doc')) and full_url.startswith(('http://iciec.isdb.org', 'https://iciec.isdb.org')) and not full_url.startswith(tuple(excluded_links)) and not "#" in full_url:
                    visited_links.add(full_url)  # Mark as visited
                    linklist.append(full_url)  # Append the new link

# Save final linklist (English URLs) to a file
with open('final_linklist.txt', 'w') as f:
    for link in linklist:
        f.write(link + '\n')

# Save excluded URLs to a file
with open('excluded_links.txt', 'w') as f:
    for link in excluded_links:
        f.write(link + '\n')

# Output the final results
print("Final linklist (English URLs):")
print(linklist)
print("Excluded linklist (Arabic and French URLs):")
print(excluded_links)
