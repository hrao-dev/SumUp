import streamlit as st
from transformers import pipeline
from PIL import Image
from news_auto_summary import NewsSummarization


def main():
    st.title(""" Sum Up! Stay Informed, Instantly """)
    st.markdown(" #### A LLM based News Summarizer App")

# Load model
    summarizer = pipeline("summarization", model="news-summary-finetuned") 

#Create header

    st.write("Sum Up! condenses the latest headlines from trusted news sources into clear, concise and easy-to-read summaries, so you can stay informed in seconds.")
    #st.write("This app is designed to help you save time by providing quick summaries of lengthy articles, making it easier to stay updated with the news.")
    #st.write("It is built leveraging a pre-trained and fine-tuned model that is capable of understanding the context and key points of the article, allowing it to generate concise and informative summaries that capture the essence of the original text.")


# Image
#image = Image.open('newspaper.jpeg')
#st.image(image)

    if st.button('News Now'):
        summary_df = NewsSummarization.get_news()
        #st.dataframe(summary_df, use_container_width=True)
        st.markdown('#### Top Stories - A Snapshot ')
        st.caption(f'**Source: Associated Press**')
        for row in summary_df.itertuples():
            # Display each article's title, summary, and URL
            container = st.container(border=True)
            container.write(f'**{row.Title}**')
            container.write(row.Summary)
            container.write(row.URL)

# Create and name sidebar for user input 
    st.sidebar.subheader("Paste any article or text below to get a quick summary:")
    input_text = st.sidebar.text_area(label="Input text:", height=200, label_visibility="collapsed")
    # Summarize button
    if st.sidebar.button("Summarize"):
        if input_text:
            # Generate the summary
            summary = summarizer(input_text,max_length=512,do_sample=False)
            st.markdown('#### Original Text ')
            # Display the original text
            #st.text_area(label ="",value=input_text, height=200)
            container = st.container(border=True)
            container.write(input_text)
            st.markdown('---')   
            # Display the summary
            st.markdown('#### Abstractive Summary ')
            st.text_area(label ="",value=summary[0]["summary_text"], height=100)
        else:
            st.warning("Enter text to summarize.")

if __name__ == "__main__":
    main()