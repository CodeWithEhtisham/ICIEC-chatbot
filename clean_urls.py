from urllib.parse import urlparse, urlunparse

# Function to normalize URLs by removing extra trailing slashes
def normalize_url(url):
    parsed_url = urlparse(url)
    # Rebuild URL without trailing slashes and double slashes
    normalized_url = urlunparse(parsed_url._replace(path=parsed_url.path.rstrip('/')))
    return normalized_url

# Read URLs from a txt file
input_file = 'final_linklist.txt'  # Your input file name

with open(input_file, 'r') as f:
    # Read all lines and strip any surrounding whitespace
    urls = [line.strip() for line in f.readlines()]

# Normalize all URLs in the list
normalized_urls = [normalize_url(url) for url in urls]

# Remove duplicates by converting the list to a set, then back to a list
unique_urls = list(set(normalized_urls))

# Optionally, sort the list to make it easier to view
unique_urls.sort()

# Output the unique normalized URLs
print("Unique normalized URLs:")
for url in unique_urls:
    print(url)

# Save the cleaned list of unique URLs to a file
output_file = 'cleaned_normalized_urls.txt'  # Your output file name
with open(output_file, 'w') as f:
    for url in unique_urls:
        f.write(url + '\n')
