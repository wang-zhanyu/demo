import streamlit as st
import time
from inference import Inference, generate_text
from PIL import Image
# from streamlit_extras.add_vertical_space import add_vertical_space

# streamlit run app_v2.py --server.fileWatcherType none
# openai.API key
model = "text-davinci-003"
api_key = "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" #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
path = "/home/zhanyu_wang/data/iu_xray/images/CXR2501_IM-1027/0.png"
st.set_page_config(
    page_title="MedChatGPT",
    page_icon=":smile:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': 'https://wang-zhanyu.github.io/',
        # 'Report a bug': "https://wang-zhanyu.github.io/",
        'About': "# This is a demo of a multimodal medical Q&A system ~"
    }
)
pneu_report = "Findings: The chest X-ray reveals findings consistent with pneumonia. The lung tissue appears hazy and white, indicating fluid accumulation in the lung tissue. The lung markings are less distinct, and the affected lung appear larger than normal. Impression: The chest X-ray findings are consistent with pneumonia, but further evaluation may be necessary to confirm the diagnosis and determine the cause and extent of the pneumonia. Additional imaging studies, laboratory tests, and a physical examination may be necessary to complete the diagnostic evaluation. Conclusion: The patient's chest X-ray findings are concerning for pneumonia and prompt further evaluation is recommended to confirm the diagnosis and determine the appropriate course of treatment."

# sidebar
with st.sidebar:
    st.subheader('Get started')
    st.write(":tada: This is a demo of a multimodal medical image Q&A system developed by the Medical Computer Vision Laboratory of the University of Sydney.")
    st.write(":memo: Upload a Chest x-ray image and input any question, our system will respond based on the observations of the image. A higher temperature setting will result in more diverse and creative outputs. The maximum length can be set to control the length of the generated response of our system.")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
    max_token = st.slider('Maximum length', 0, 256, 128)
    # num_beams = st.slider('Beam Search', 1, 5, 3)


st.write("<style>h1{text-align:center;}</style>", unsafe_allow_html=True)
st.title(":male-doctor: Medical image Q&A system")
# add_vertical_space(2)

option = st.sidebar.selectbox(
    "Use an example image if you don't have one",
    ('Upload', 'Use example'))


if option == "Upload":
    uploaded_file = st.file_uploader("upload a chest-xray image", help="Right now, you can only upload Chest-xray images.")
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        with st.spinner("Please wait while the model is analyzing the image"):
            gen_report, label, score, detect = Inference(uploaded_file, num_beams=3)

        if "radiogram" not in label:
            st.warning(f"Please input an Chest x-ray image.")
        else:
          prompt = st.text_input(label="You: ", placeholder="input your question here")
          if prompt:
            #   st.write(detect['Pneumonia'])
              with st.spinner("Please wait while the answer is being generated"):
                report = pneu_report if detect['Pneumonia'] > 0.5 else gen_report
                input_prompt = f"I want you to play the role of an AI-assisted radiologist. I will provide you with a chest x-ray image and ask you a question. Your task will be to use the relevant medical knowledge, to answer the input questions. In this chest x-ray image: {report}. My question is: {prompt}. You should answer the question in the same language as the question."
              
                try:
                    generated_text = generate_text(input_prompt, model, api_key, temperature=temperature, max_tokens=max_token)
                    if generated_text != "":
                        st.text_area("Respond:", generated_text, height=250)
                except Exception as e:
                    st.write('Please try again.')
    
    
if option == 'Use example':
    st.image(path, use_column_width=True)
    report, _, _, detect = Inference(path, num_beams=3)
    prompt = st.text_input(label="You: ", placeholder="input your question here")
    if prompt:
        with st.spinner("Please wait while the answer is being generated"):
            input_prompt = f"I want you to play the role of an AI-assisted radiologist. I will provide you with a chest x-ray image and ask you a question. Your task will be to use the relevant medical knowledge, to answer the input questions. In this chest x-ray image: {report}. My question is: {prompt}. You should answer the question in the same language as the question."
            try:
                generated_text = generate_text(input_prompt, model, api_key, temperature=temperature, max_tokens=max_token)
                if generated_text != "":
                    st.text_area("Respond:", generated_text, height=250)
            except Exception as e:
                st.write('Please try again.')
