import requests
from bs4 import BeautifulSoup
import csv

# Define the URL to scrape
url = "https://www.invajy.com/sad-quotes/"

# Send a GET request to the URL
response = requests.get(url, headers={"User-Agent": "Roku4640X/DVP-7.70 (297.70E04154A)"})
print(response)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find the sad text elements in the HTML
sad_text_elements = soup.find_all("li")
print(sad_text_elements)
sad_texts = [element.text for element in sad_text_elements]

# Save the sad texts to a CSV file
with open("sad_texts.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for text in sad_texts:
        writer.writerow([text, 1])
