import json
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

# Define the configuration for the scraping pipeline
graph_config = {
    "llm": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "openai/gpt-4o-mini",
    },
    "verbose": True,
    "headless": False,
}

# Create the SmartScraperGraph instance
smart_scraper_graph = SmartScraperGraph(
    prompt="list down all faqs",
    source="https://iciec.isdb.org/",
    config=graph_config
)

# Run the pipeline
result = smart_scraper_graph.run()
print(json.dumps(result, indent=4))