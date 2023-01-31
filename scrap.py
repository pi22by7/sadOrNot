import requests
from bs4 import BeautifulSoup
import csv

# Define the URL to scrape
url = "https://twitter.com/search?q=depression"

# Send a GET request to the URL
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, "
                                                    "like Gecko) Chrome/47.0.2526.111 Safari/537.36"})
print(response)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find the sad text elements in the HTML
sad_text_elements = soup.find_all("div", class_="css-901oao")
print(sad_text_elements)
sad_texts = [element.text for element in sad_text_elements]

# Save the sad texts to a CSV file
with open("sad_texts.csv", "a", encoding='utf-8') as f:
    writer = csv.writer(f)
    # writer.writerow(["text", "label"])
    writer.writerow([])
    for text in sad_texts:
        writer.writerow([text, 1])
