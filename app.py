import gradio as gr
import requests
from newspaper import Article
from transformers import pipeline
import config
import nltk
import os
import PyPDF2


# Load summarization pipeline
summarizer = pipeline("summarization", model="harao-ml/flant5-finetuned-summarize")

# Function to split text into smaller chunks
def split_text(text, max_tokens=512):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

# Function to clean text
def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text


# Helper function to fetch and parse an article from a URL
def fetch_article_details(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        title = article.title or "Untitled"
        author = ", ".join(article.authors) if article.authors else "Unknown"
        pub_date = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "Unknown"
        return title, author, pub_date, article.text
    except Exception as e:
        return None, None, None, f"Error fetching article: {str(e)}"

# Helper function to generate a summary
def generate_summary(content):
    if not content.strip():
            return "No input provided."
    text = content
    cleaned_text = clean_text(text)
    chunks = list(split_text(cleaned_text))
    cons_summary = ''.join([summarizer(chunk, do_sample=False)[0]['summary_text'] for chunk in chunks if chunk.strip()]) if chunks else ''
    summary = summarizer(text, do_sample=False)[0]['summary_text']
    return cons_summary

# Summarize from text or URL
def summarize_input(mixed_input):
    if mixed_input.startswith("http://") or mixed_input.startswith("https://"):
        title, author, pub_date, content = fetch_article_details(mixed_input)
        if content.startswith("Error"):
            return f"### Error\n\n{content}"
        summary = generate_summary(content)
        return f"**Title:** {title}\n\n**Author(s):** {author}\n\n**Published:** {pub_date}\n\n**📝 Summary** \n\n{summary}\n\n[🔗 Read more]({mixed_input})\n\n---"
    else:
        summary = generate_summary(mixed_input)
        return f"## 📝 Summary \n\n{summary}\n\n📎 **Original Text:**\n\n{mixed_input}\n\n---"
    
# Function to summarize a file (PDF or TXT)
def summarize_file(file):
    try:
        if file is None:  # Handle the case where no file is provided
            return ""  # Return an empty string instead of an error message

        text = ""
        if file.name.endswith(".pdf"):
            with open(file.name, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif file.name.endswith(".txt"):
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return "❌ Unsupported file type."

        if not text.strip():
            return "❌ No text found in file."

        summary = generate_summary(text)
        original_text = text

        # Combine the outputs into a single string
        result = (
            f"### 📝 Summary\n\n"
            f"{summary}\n\n"
            f"---\n\n"
            f"📎 **Original Extracted Text:**\n\n{original_text}"
        )
        return result
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"



# Function to fetch top headlines from NewsAPI and summarize them
def fetch_news():
    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': config.api_key,
        'language': 'en',
        'sources': 'associated-press',
        'pageSize': 10
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Error: Failed to fetch news. Status code: {response.status_code}"

        articles = response.json().get("articles", [])
        summaries = [f'## 📰 Top Stories - Instant Insights\n\n']
        for article in articles:
            title = article.get("title", "No title")
            article_url = article.get("url", "#")
            author = article.get("author", "Unknown")
            pub_date = article.get("publishedAt", "Unknown")
            content = extract_full_content(article_url) or article.get("content") or article.get("description") or ""
            summary = generate_summary(content)
            summaries.append(f"**{title}** \n\n**Author(s):** {author}\n\n**Published:** {pub_date}\n\n**📝 Summary:** {summary}\n\n [🔗 Read more]({article_url})\n\n---")

        if not summaries:
            return "### No articles could be summarized."
        return "\n\n".join(summaries)
    except Exception as e:
        return f"### Error fetching news\n\n{str(e)}"

# Helper function to extract full content using newspaper3k
def extract_full_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None

# Gradio interface
with gr.Blocks(theme=gr.themes.Default(font="Arial", font_mono="Courier New")) as demo:
    # Header Section
    gr.Markdown("# 📰 Sum Up! Stay Informed, Instantly")
    gr.Markdown("### FLAN-T5-Driven Summarizer for Multi-Format Content")
    gr.Markdown(
        "Sum Up! effectively distills lengthy content into clear, concise summaries with just a text input, file upload, or URL. Stay informed with instant access to auto-summarized top news headlines—all in just one click.")

    # Input Section
    gr.Markdown("---")  # Horizontal line for separation
    with gr.Row():
        # Left Column: Collapsible Sidebar for Latest News
        with gr.Column(scale=1, min_width=300):
            with gr.Accordion("📢 News at a Glance", open=False):
                gr.Markdown("**Source: Associated Press**")
                gr.Markdown("Click to get today's top news from the Associated Press, simplified and ready to read")
                news_btn = gr.Button("⚡ News Now", variant="primary", elem_id="news-now-btn")

        # Right Column: Text Input and File Upload
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### Provide content to summarize")
            gr.Markdown("#### Enter Text or URL")
            input_box = gr.Textbox(
                label="",
                placeholder="Paste a URL or text here...",
                lines=5,
            )
            summarize_btn = gr.Button("🔍 Summarize", variant="primary", elem_id="summarize-btn")

            # Clear Button placed below the Summarize button
            clear_btn = gr.Button("Clear", variant="secondary", elem_id="clear-btn")

            gr.Markdown("#### Upload a File")
            file_input = gr.File(
                label="Upload a .pdf or .txt file", file_types=[".pdf", ".txt"]
            )
            gr.Markdown("**Note:** Only PDF and TXT files are supported.")

    # Output Section
    gr.Markdown("---")  # Horizontal line for separation
    gr.Markdown("### 💡 Key Takeaways")
    with gr.Row():
        with gr.Column(scale=1):
            gen_output = gr.Markdown()  # Use a valid output component

    # Link buttons to their respective functions
    summarize_btn.click(fn=summarize_input, inputs=input_box, outputs=gen_output)
    file_input.change(fn=summarize_file, inputs=file_input, outputs=gen_output)
    news_btn.click(fn=fetch_news, inputs=[], outputs=gen_output)

    # Clear button functionality
    clear_btn.click(
        fn=lambda: ("", None, ""),  # Clear all inputs and outputs
        inputs=[],
        outputs=[input_box, file_input, gen_output],
    )

# Ensure gen_output is properly reset
gen_output = gr.Markdown(value="")  # Initialize with an empty value

# Add custom CSS for better styling
css = """
#summarize-btn {
    background-color: #4CAF50 !important; /* Green for Summarize */
    color: white !important;
    font-size: 16px !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    margin-top: 20px !important;
    width: 100%;
}

#news-now-btn {
    background-color: #0078D7 !important; /* Blue for News Now */
    color: white !important;
    font-size: 16px !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    margin-top: 20px !important;
    width: 100%;
}

#clear-btn {
    background-color: #d6d8db !important; /* Lighter Gray for Clear */
    color: black !important;
    font-size: 16px !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    margin-top: 20px !important;
    width: 100%;
}
"""

# Apply the custom CSS
demo.css = css

if __name__ == "__main__":
    demo.launch()