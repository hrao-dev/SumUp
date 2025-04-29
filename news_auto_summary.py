# Import necessary libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline

# pd.set_option('display.max_colwidth', 1000)

class NewsSummarization():
    """
    A class to summarize news articles using the News API and Hugging Face Transformers.
    """

    def __init__():
        """
        Initializes the NewsSummarization class.
        """
        pass
    def get_news():
        #self.summarizer = pipeline("summarization", model="news-summary-finetuned")

        # Setting up the News API Request

        # API key
        api_key = 'e3eca7fe616e4895bbd8c162ec9d567d'

        # Specify the endpoint for top headlines from Associated Press
        url = 'https://newsapi.org/v2/top-headlines?'

        # Parameters for the request
        params = {
            #'q': 'technology',        # query keyword
            'apiKey': api_key,         # API key
            'language': 'en',          # search only for English articles
            'sources': 'associated-press',  # specify sources
            'pageSize': 20             # limit the number of articles returned to 10
        }

        # Fetching the news articles and extracting the content

        # Make the request to the News API
        response = requests.get(url, params=params)

        # Function to extract main content from HTML
        def extract_content(url):
            try:
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')

                # This part needs to be adapted to the structure of the target websites
                paragraphs = soup.find_all('p')
                content = ' '.join([para.get_text() for para in paragraphs])

                return content if content else 'Content not available'
            except Exception as e:
                print(f"Failed to retrieve content from {url}. Error: {e}")
                return 'Content not available'
    
        # Splitting Text for Summarization

        # Function to split text into smaller chunks
        def split_text(text, max_tokens=512):
            words = text.split()
            for i in range(0, len(words), max_tokens):
                yield ' '.join(words[i:i + max_tokens])


        # Summarization pipeline from Hugging Face

        summarizer = pipeline("summarization",model="news-summary-finetuned")

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles')[:10]  # Limit the results to 10 articles

            # Extract relevant information and store it in a list of dictionaries
            articles_list = []
            for article in articles:
                full_content = extract_content(article['url'])

                # Summarize the content if it's available
                if full_content != 'Content not available':
                    #chunks = list(split_text(full_content))
                    #summary = ' '.join([summarizer(chunk, do_sample=False)[0]['summary_text'] for chunk in chunks])
                    summary = summarizer(full_content, do_sample=False)[0]['summary_text']
                else:
                    summary = 'Unable to Summarize as content is not available'

                articles_list.append({
                    'Title': article['title'],
                    #'Content': full_content,
                    'Summary': summary,
                    'URL': article['url'],
                    #'Source': article['source']['name'],
                    'Published_At': article['publishedAt'],

            })

            # Create a pandas DataFrame from the list of dictionaries
            df = pd.DataFrame(articles_list)
            print(df)
            # Save the DataFrame to a CSV file
            df.to_csv('latest_news.csv', index=False)
            return df
        else:
            print(f"Failed to retrieve articles. Status code: {response.status_code}")