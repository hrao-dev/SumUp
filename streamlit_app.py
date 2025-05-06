import streamlit as st
import requests
import os
import config
import re
from dotenv import load_dotenv

#load_dotenv()


# Set up the Streamlit app
st.set_page_config(page_title="Sum Up!", page_icon=":newspaper:", layout="wide")
st.title(""" Sum Up! Stay Informed, Instantly """)
st.markdown(" #### A LLM based News Summarizer App")
    

# Add a brief description
st.write("Sum Up! condenses the latest headlines from trusted news sources into clear, concise and easy-to-read summaries, so you can stay informed in seconds.")
st.write("Simply paste a news article URL or some text, and let the app generate a summary for you. You can also get the latest news headlines from Associated Press, summarized for your convenience.")

# Define the API endpoint for summarization
API_ENDPOINT = "http://localhost:8000/summarize/"
api_key = config.api_key # Replace with actual NewsAPI key


 # Specify the endpoint for top headlines from Associated Press
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines?'

# Parameters for the request
params = {
            #'q': 'technology',        # query keyword
            'apiKey': api_key,         # API key
            'language': 'en',          # search only for English articles
            'sources': 'associated-press',  # specify sources
            'pageSize': 20             # limit the number of articles returned to 10
        }

# Fetch top headlines from AP and auto summarize
if st.button("📰 News Now"):
    if not api_key:
        st.error("Missing NewsAPI key.")
    else:
        with st.spinner('Fetching latest news...'):
            news_res = requests.get(NEWS_API_URL, params=params)
            if news_res.ok:
                articles = news_res.json().get("articles", [])[:10]  # Limit to top 10 articles
                st.markdown('#### Top Stories - A Snapshot ')
                st.caption(f'**Source: Associated Press**')
                for i, article in enumerate(articles, 1):
                    title = article["title"]
                    url = article["url"]
                    container = st.container(border=True)
                    #st.markdown(f"##### {title}")
                    container.write(f'**{title}**')
                    # Send URL to backend for summarization
                    sum_payload = {"text": url, "type": "url"}
                    sum_res = requests.post(API_ENDPOINT, json=sum_payload)
                    if sum_res.ok:
                        summary = sum_res.json().get("summary", "No summary.")
                        #st.markdown(f"**Summary:** {summary}")
                        container.write(f'**Summary:** {summary}')
                    else:
                        st.error("⚠️ Failed to summarize article.")
                    #st.markdown(f"[🔗 Read full article]({url})")
                    container.write(f"[🔗 Read full article]({url})")
            else:
                st.error("Could not fetch top news.")


# Summarize User Input

# Regex to check for URL
def is_url(text):
    return re.match(r'^https?://', text.strip()) is not None

# Unified input box
user_input = st.sidebar.text_area(f"**Paste text or a news article URL here:**", height=100)

if st.sidebar.button("🔍 Summarize"):
    if not user_input.strip():
        st.warning("Please enter text or a valid URL.")
    else:
        input_type = "url" if is_url(user_input.strip()) else "text"
        payload = {"text": user_input.strip(), "type": input_type}
        with st.spinner("Summarizing..."):
            res = requests.post(API_ENDPOINT, json=payload)
            if res.ok:
                summary = res.json().get("summary")
                input_text = res.json().get("original_content")
                title = res.json().get("title")
                authors = res.json().get("authors")
                published_date = res.json().get("published_date")
                st.success("Summary generated successfully!")
                if input_type == "url":
                    # Display article metadata
                    st.write(f"**Title:** {title}")
                    st.write(f"**Authors:** {', '.join(authors) if authors else 'Unknown'}")
                    st.write(f"**Published Date:** {published_date if published_date else 'Unknown'}")
                # Display the summary
                st.subheader("📝 Summary")
                st.write(summary)
                    
                # Display original text (collapsed)
                with st.expander("Show original article"):
                    st.write(input_text)
            else:
                st.error(res.json().get("error", "Something went wrong."))
