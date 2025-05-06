# FastAPI application for summarizing text or articles from URLs

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from newspaper import Article
import os
from dotenv import load_dotenv

#load_dotenv()

app = FastAPI()

# Initialize the summarization pipeline
summarizer = pipeline("summarization",model="news-summary-finetuned")

class TextRequest(BaseModel):
    text: str
    type: str  # "text" or "url"

@app.post("/summarize/")
async def summarize(request: TextRequest):
    content = ""
    if request.type == "url":
        article = Article(request.text)
        try:
            article.download()
            article.parse()
            content = article.text
            title = article.title
            authors = article.authors
            published_date = article.publish_date
        except Exception as e:
            return {"error": f"Failed to process article: {str(e)}"}
    else:
        content = request.text
    if not content:
        return {"error": "No content to summarize."}
    
    # Function to split text into smaller chunks
    def split_text(text, max_tokens=512):
            words = text.split()
            for i in range(0, len(words), max_tokens):
                yield ' '.join(words[i:i + max_tokens])
    
    # Split content into manageable chunks
    chunks = list(split_text(content))


    # Generate summary using the summarization pipeline
    cons_summary = ''.join([summarizer(chunk, do_sample=False)[0]['summary_text'] for chunk in chunks if chunk.strip()]) if chunks else ''
    #summary = summarizer(content, do_sample=False)[0]['summary_text']

    return {"summary": cons_summary, "original_content": content, "title": title if request.type == "url" else None,
            "authors": authors if request.type == "url" else None,
            "published_date": published_date if request.type == "url" else None}    

