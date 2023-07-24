from langchain import PromptTemplate
from langchain.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


#markdown
from markdownify import markdownify as md
# streamlit
import streamlit as st

# Scraping tools
from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md

# Yt
from langchain.document_loaders import YoutubeLoader

# envs
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

# Load the llm

def LLM():
    #co = cohere.Client(os.environ["COHERE_API_KEY"])
    llm = Cohere(temperature=0.7, max_tokens=2000)
    return llm

# Pull data from YouTube in text form
def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript



def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap = 0,)

    docs = text_splitter.create_documents([user_information])
    return docs

response_types = {
    'QA':"""Your goal is to generate questions and followed by their answers from the text""",

    'Short Summary':"""Your goal is to generate short summary from the {text} with bullet points""",

    'Long Summary':"""Your goal is to generate long summary within 100 words from the {text}""",

    'JSON':"""Your goal is to analyze the {text} and give a json output with the following information:

        Identify the subject of the {text},
        Identify if any specifications about a product is being described in the {text},
        Identify if any names are mentioned in the {text},
        Identify if any pricing information is mentioned in the {text},
        Identify if any names are mentioned in the {text},
        Finally give the output in the json format by following the example format below
        %EXAMPLE
        ```json
        {
            "subject": YOUR_RESPONSE,
            "specs": YOUR_RESPONSE,
            "names": YOUR_RESPONSE,
            "pricing": YOUR_RESPONSE
        }```
        """
}

map_prompt = """You are a helpful AI bot with a research orientation.Below is an information about a topic. Information will include either website information or video transcripts or both about the {topic}.Your goal is to generate intuitive questions from the topic that highlights cruical informations included in the topic,followed by the answer to the question. Use specifics as possible and donot make up things.

% START OF INFORMATION about {topic}:
{text}
% END OF INFORMATION ABOUT {topic}:

Please respond with list of a few interview questions based on the topics above followed by their answers,use the following example format-

% START OF THE EXAMPLE FORMAT:
\n\n
Q- Frame the question here with proper punctuation.\n

--------------------------------------------------

A - Frame the answer to the above question here within 50 words.
% END OF THE EXAMPLE FORMAT:
\n\n

YOUR RESPONSE:"""


map_prompt_template = PromptTemplate(template=map_prompt,input_variables=['text','topic'])

combine_prompt = """You are a helpful AI bot, You will be give a list of potential interview questions along with their answers that we can ask about the {topic}

Please consolidate the questions and answers and return a list

% PROBABLE QUESTIONS AND ANSWERS

\n\n
{text}
\n\n
% END OF PROBABLE QUESTIONS AND ANSWERS

YOUR RESPONSE:
"""

combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text','topic'])


# work on streamlit starts here

st.set_page_config(page_title="LLM assisted Youtube QA/Summary",page_icon=':leaf:')

st.header("LLM assisted Youtube QA/Summarization")

col1, col2 = st.columns(2)

#Start of explaining the columns in streamlit
with col1:
    st.markdown("Fire away your youtube video link, and generate questions and answers or even summarize the whole video if you are in a hurry. WHO ELSE ARE NOT IN A HURRY THESE DAYS ?! :smile:")

with col2:
    st.image('/home/susearc/Documents/github/LLM_QA_STREAMLIT/summarization-img.png',caption="summarization")
# End of the columns

st.markdown('## :teacher: Your personal bot')

output_type = st.radio(
    "Output_type:",('QA','Short Summary','Long Summary','JSON')
)

topic = st.text_input("Topic",placeholder='Name of the topic',key='topic')
yt_videos = st.text_input("Youtube video link",placeholder="Ex: https://www.youtube.com/watch?v=45ETZ1xvHS0",key='yt_video')


# Output
st.markdown(f'### {output_type}:')

button_ind = st.button(label='Generate Output',type='primary')


# Check if the button clicked
if button_ind:
    if not (yt_videos or topic):
            st.warning('Please provide a valid link')
            st.stop()

    video_text = get_video_transcripts(yt_videos) if yt_videos else ""

    # start the llm process
    llm = LLM()
    user_docs = split_text(video_text)
    if output_type == 'QA':

        chain = load_summarize_chain(llm,
                                chain_type='map_reduce',
                                map_prompt = map_prompt_template,
                                combine_prompt =   combine_prompt_template,
                                )

        st.write("Sending to LLM ...")
        st.write("Hold on tight ... this might take a while..")
        output = chain({'input_documents':user_docs,'text':video_text,'topic':topic},return_only_outputs=True)

        st.markdown(f"## Output:")
        st.write(md(output['output_text']))
    
    else:
        templ = PromptTemplate(template=response_types.get(output_type),input_variables=['text'])
        output = llm(prompt=templ.format(text=user_docs))


# if the user wants to see the output

        
        st.write("Hold on tight it might take a while .....")
        st.write("Sending to LLM ...")

        st.markdown(f"## Output:")
        st.write(md(output))


st.info("This is a project made by ARKA PRAVA PANDA with :hearts:")

#commit check