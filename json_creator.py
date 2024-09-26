import json

# Function to convert the text file into a structured list of dictionaries
def convert_txt_to_json(txt_file_path, json_file_path):
    # Initialize an empty list to hold the data
    data = []

    # Read the text file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()

        # Split the content by the URL separator (assuming "================================================================================" is the separator)
        entries = content.split('================================================================================')

        for entry in entries:
            # Clean up each entry
            entry = entry.strip()
            if entry:  # Ensure the entry is not empty
                # Split entry into URL and content
                lines = entry.split('\n')
                url = lines[0].replace("URL:", "").strip()
                # Combine the rest of the lines as content
                content = "\n".join(line.strip() for line in lines[1:] if line.strip())
                
                # Append the structured data as a dictionary
                data.append({
                    "url": url,
                    "content": content
                })

    # Write the structured data to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# Define file paths
txt_file_path = 'scraped_data_with_metadata.txt'  # Change this to your input text file path
json_file_path = 'clean_data.json'  # This will be the output JSON file

# Convert the text file to JSON
convert_txt_to_json(txt_file_path, json_file_path)

print(f"Converted {txt_file_path} to {json_file_path} successfully!")
