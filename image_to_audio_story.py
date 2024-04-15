# converts provided text to an audio story
# 1) uses Salesforce/blip-image-captioning-base (Huggingface) to caption image
# 2) uses Google Gemini to generate story on the above caption
# 3) uses suno/bark-small (Huggingface) to create an audio file
# 4) UI using streamlit
# requires Huggingfacehub and Google generative AI API tokens

from IPython.display import Audio
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import streamlit as st
from transformers import pipeline

import requests

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token
# os.environ["GOOGLE_API_KEY"] = google_api_key


# image-to-text
# check huggingface.co/tasks for the source of the task name "image-to-text"
def img_to_text(url):
    image_to_text = pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-base",
        max_new_tokens=200,
    )
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text


# llm
def generate_story(scenario, google_api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    template = """
    You are a storyteller;
    you can generate a short story based on a simple narrative;
    the story should be no more than 50 words long;
    
    Context: {scenario}
    Story:"""

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    story = chain.invoke(scenario)
    return story["text"]


# text-to-speech
def text_to_speech(message):
    txt_to_speech = pipeline(task="text-to-speech", model="suno/bark-small")
    speech = txt_to_speech(message)
    print(speech)

    sound = Audio(speech["audio"], rate=speech["sampling_rate"])
    with open("output/audio.flac", "wb") as file:
        file.write(sound.data)


# alternatively, implementing using the Inference API
def text_to_speech2(message, huggingfacehub_api_token):
    API_URL = "https://api-inference.huggingface.co/models/suno/bark-small"
    headers = {"Authorization": f"Bearer {huggingfacehub_api_token}"}
    payloads = {"text_inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads)
    print(response)

    with open("output/audio.mp4", "wb") as file:
        file.write(response.content)


# UI on streamlit
def main():
    st.set_page_config(page_title="Image to audio story", page_icon="🎧")
    st.header("Convert an image into an audio story")
    image_file = st.file_uploader("Choose an image", type="jpg")

    google_api_key = st.sidebar.text_input("Input you Google generative AI API key")
    # huggingfacehub_api_token = st.sidebar.text_input(
    #     "Input your Hugging Face Hub API token"
    # )

    if google_api_key and image_file is not None:
        print(image_file)
        image_bytes = image_file.getvalue()
        with open(image_file.name, "wb") as file:
            file.write(image_bytes)
        scenario = img_to_text(image_file.name)
        story = generate_story(scenario, google_api_key)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("output/audio.flac")


if __name__ == "__main__":
    main()
