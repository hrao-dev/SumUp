
# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import newspaper
import torch
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
from urllib.parse import urlparse

# Initialize session state for model and tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource

def get_news():
    """Fetch latest news articles from Associated Press and summarize them."""
    with st.spinner("Fetching latest news..."):
            
 
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
                if full_content:
                    chunks = list(split_text(full_content))
                    summary = ''.join([summarizer(chunk, do_sample=False)[0]['summary_text'] for chunk in chunks if chunk.strip()]) if chunks else ''
                    #summary = summarizer(full_content, do_sample=False)[0]['summary_text']
                else:
                    summary = 'Unable to Summarize as content is not available'

                articles_list.append({
                    'Title': article['title'],
                    'Content': full_content,
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

def clean_text(text):
    """Clean and preprocess the input text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove very long words (likely garbage)
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def summarize_text(text):
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Load Model and Summarization pipeline
        summarizer = pipeline('summarization', model = 'news-summary-finetuned')
        
        summary = summarizer(text, do_sample=False)[0]['summary_text']
        return summary
    
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None

def fetch_article(url):
    """Fetch article content and metadata from URL using newspaper3k"""
    try:
        # Download and parse the article
        article = newspaper.Article(url)
        
        # Enable extraction of all possible metadata
        article.download()
        article.parse()
        
        # Extract metadata
        title = article.title or 'No title found'
        authors = ', '.join(article.authors) if article.authors else 'No author information'
        publish_date = article.publish_date or 'No publish date found'
        
        # Extract publisher from URL domain
        publisher = urlparse(url).netloc.replace('www.', '').capitalize() or 'No publisher information'
        
        # Get the main text content
        text = article.text or ''
        
        return title, authors, str(publish_date), publisher, text
    
    except Exception as e:
        st.error(f"Error fetching the article: {str(e)}")
        return None, None, None, None, None
    
def display_summary(summary,text):
    if summary:
                st.success("Summary generated successfully!")
                st.write("### Summary")
                st.write(summary)
                    
                # Display original text (collapsed)
                with st.expander("Show original article"):
                    st.write(text)

    
def main():
    # Set up the Streamlit app
    st.set_page_config(page_title="Sum Up!", page_icon=":newspaper:", layout="wide")
    st.title(""" Sum Up! Stay Informed, Instantly """)
    st.markdown(" #### A LLM based News Summarizer App")
    
    # Load model
    summarizer = pipeline("summarization", model="news-summary-finetuned") 

    # Create header
    st.write("Sum Up! condenses the latest headlines from trusted news sources into clear, concise and easy-to-read summaries, so you can stay informed in seconds.")
    #st.write("This app is designed to help you save time by providing quick summaries of lengthy articles, making it easier to stay updated with the news.")
    #st.write("It is built leveraging a pre-trained and fine-tuned model that is capable of understanding the context and key points of the article, allowing it to generate concise and informative summaries that capture the essence of the original text.")


    # Display latest news

    if st.button('News Now'):
        summary_df = get_news()
        #st.dataframe(summary_df, use_container_width=True)
        st.markdown('#### Top Stories - A Snapshot ')
        st.caption(f'**Source: Associated Press**')
        for row in summary_df.itertuples():
            # Display each article's title, summary, and URL
            container = st.container(border=True)
            container.write(f'**{row.Title}**')
            #container.write(row.Content)
            container.write(row.Summary)
            container.write(row.URL)
    
    # Summarize User Input
    input_text = st.sidebar.text_input("Paste any article text or URL below to get a quick summary:")
    
    if st.sidebar.button("Summarize"):
        if input_text:
            # Check if the input is a URL
            if not input_text.startswith(('http://', 'https://')):
                # If it's not a URL, treat it as plain text and generate the summary
                summary = summarize_text(input_text)
                # Display the summary
                display_summary(summary, input_text)
            else:
                # Check if the input is a valid URL
                try:
                    url = input_text.strip()
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        raise ValueError("Invalid URL")
                except ValueError as e:
                    st.error(f"Invalid URL: {str(e)}")
                    return
                # If it's a valid URL, fetch the article and generate the summary
                with st.spinner("Fetching article and generating summary..."):
                
                # Fetch article
                    title, authors, publish_date, publisher, article_text = fetch_article(url)
            
                    if article_text:
                        # Display metadata
                        st.write(f"**Title**: {title}")
                        st.write(f"**Authors**: {authors}")
                        st.write(f"**Publish Date**: {publish_date}")
                        st.write(f"**Publisher**: {publisher}")
                
                        # Generate summary
                        summary = summarize_text(article_text)
                        display_summary(summary, article_text)
                    else:
                        st.error("Failed to fetch the article. Please check the URL and try again.")
        else:
            st.error("Please enter a valid article text or URL to summarize.")


if __name__ == "__main__":
    main()