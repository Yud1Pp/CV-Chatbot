from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
import streamlit as st
import groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
NOMIC_API_KEY = st.secrets["NOMIC_API_KEY"]

st.title("Yudi's AI Assistant 🤖")
language = st.sidebar.selectbox("Pilih Bahasa / Choose Language", ["Indonesia", "English"])

model = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    api_key=GROQ_API_KEY,
    reasoning_format="hidden"
)

intro = "Hello, I am a bot. How can I help you?" if language == "English" else "Halo, saya adalah bot. Ada yang bisa saya bantu?"

if "message" not in st.session_state:
    st.session_state.message = []

if "intro" not in st.session_state:
    st.session_state.intro = {
        "Indonesia" : "Halo, saya adalah bot. Ada yang bisa saya bantu?",
        "English" : "Hello, I am a bot. How can I help you?",
    }

def retrieve_docs(query, vector_store):
    return vector_store.similarity_search(query)

def answer_question(question, documents, history):
    context = "\n\n".join([doc.page_content for doc in documents])

    template_en = """
        [INST]You are an AI assistant designed to answer questions based on the CV of Yudi Pratama Putra, a male professional.  
        Use the provided context and history extracted from their CV to produce detailed, complete and accurate answers. In addition, you must answer based on the chat history with the user. For your skills and projects, please emphasize those related to Artificial Intelligence and Machine Learning. Also remember that the smart door system that you are working on is not AI and ML.

        Projects that Yudi has worked on and the time the project took place:
        • Postura (December 2024): Yudi developed an IoT system using an ESP32-CAM for 
        real-time streaming, employing OpenCV for image processing and Mediapipe for pose 
        estimation. He built two machine learning models—one to classify good versus bad 
        posture and another to detect normal posture, scoliosis, and lordosis—integrated into a 
        web application with a Flask backend for online posture monitoring.
        • Melanoma Detection (December 2024): This project focused on detecting and 
        analyzing melanoma types through image processing with OpenCV and machine 
        learning. Yudi extracted image features such as segmentation, edge detection, 
        asymmetry, and color analysis to train the model, and developed an inference interface 
        using Streamlit that allows users to upload images for melanoma detection.
        • Fire Alarm with OpenCV (December 2024): Yudi developed a Python-based fire 
        detection system utilizing OpenCV and camera input. By implementing HSV 
        thresholding, the system detects fire colors and integrates MQTT communication 
        through a Mosquitto broker, with the ESP32 programmed to activate a warning light 
        upon detection.
        • Kard AI (November 2024): He developed a web-based platform called Kard AI for 
        the early detection of cardiovascular disease risk using a Random Forest model. The 
        platform also features a fine-tuned FLAN-T5 chatbot designed to educate users on 
        cardiovascular health and risk factors, providing personalized risk assessments and 
        educational content.
        • Maize Pest Detection (October 2024): Yudi built a YOLOv11-based model to detect 
        pests in maize using a filtered IP102 dataset. His model achieved an mAP of 84.3%, 
        precision of 87.6%, and recall of 79.4%, enabling real-time targeted pest control to 
        support sustainable farming practices.
        • Emotion Classification (August 2024): He developed an emotion classification model 
        using the FastJobs/Visual Emotional Analysis dataset. By fine-tuning a Google-ViT 
        model, Yudi achieved an accuracy of 63.75% with a loss of 1.0965. The model has been 
        downloaded 237 times, indicating its value and effectiveness.
        • Gesture-Controlled Robot (August 2024): In this project, Yudi developed a robotic 
        hand controlled by hand gestures using an Arduino microcontroller and a servo-thread 
        mechanism. He integrated OpenCV and Mediapipe for real-time hand tracking, where 
        detected finger angles were used to precisely control the servo motors.

        - Answer the question in detail, it must be very detailed, but if the question is simple and specific answer it simply.
        - If the question is unclear, assume it refers to his professional background, skills, or experience.  
        - If you don't know the answer, don't make it up, just say sorry I don't know.

        Question: {question}  
        Context: {context}  
        [/INST]
    """

    template_id = """
        [INST]Anda adalah asisten AI yang dirancang untuk menjawab pertanyaan berdasarkan CV Yudi Pratama Putra, seorang profesional pria.
        Gunakan konteks dan history yang disediakan yang diekstrak dari CV-nya untuk menghasilkan jawaban yang detail, lengkap dan akurat. Selain itu, anda harus menjawab berdasarkan history chat dengan user. Untuk skill dan proyek yang Yudi punya, tolong tonjolkan yang berhubungan dengan Artificial Intelligence dan Machine Learning. Ingat juga bahwa smart door system yang yudi kerjakan bukanlah AI dan ML.

        Proyek yang pernah Yudi kerjakan serta waktu proyek berlangsung:
        • Postura (Desember 2024): Yudi mengembangkan sistem IoT berbasis ESP32-CAM 
        untuk streaming secara real-time, menggunakan OpenCV dan Mediapipe untuk 
        estimasi pose. Ia juga membangun dua model machine learning, satu untuk 
        mengklasifikasikan postur baik dan buruk, serta satu lagi untuk mendeteksi postur 
        normal, skoliosis, dan lordosis. Hasilnya diintegrasikan ke dalam sebuah aplikasi web 
        dengan backend Flask untuk monitoring postur secara online.
        • Melanoma Detection (Desember 2024): Proyek ini berfokus pada deteksi dan analisis 
        jenis melanoma melalui image processing menggunakan OpenCV dan machine 
        learning. Yudi mengekstraksi fitur gambar seperti segmentasi, edge detection, asimetri, 
        dan analisis warna untuk melatih model, serta menciptakan antarmuka inferensi 
        berbasis Streamlit untuk memungkinkan pengguna mengunggah gambar dalam proses 
        deteksi.
        • Fire Alarm with OpenCV (Desember 2024): Dalam proyek ini, Yudi 
        mengembangkan sistem deteksi kebakaran dengan menggunakan Python dan OpenCV. 
        Teknik thresholding pada ruang warna HSV diterapkan untuk mendeteksi api, 
        kemudian sistem ini diintegrasikan dengan komunikasi MQTT melalui broker 
        Mosquitto, dengan ESP32 yang diprogram untuk mengaktifkan lampu peringatan saat 
        kebakaran terdeteksi.
        • Kard AI (November 2024): Yudi mengembangkan sebuah platform berbasis web 
        untuk deteksi dini risiko penyakit kardiovaskular menggunakan model Random Forest. 
        Ia juga mengintegrasikan chatbot berbasis FLAN-T5 yang telah disesuaikan untuk 
        memberikan edukasi mengenai kesehatan jantung dan faktor risiko, sehingga platform 
        ini dapat menyediakan penilaian risiko dan konten edukatif secara personal.
        • Maize Pest Detection (Oktober 2024): Dalam proyek ini, Yudi membangun model 
        deteksi hama pada tanaman jagung menggunakan YOLOv11 dan dataset IP102 yang 
        telah disaring. Model yang dibuat berhasil mencapai mAP 84.3%, presisi 87.6%, dan 
        recall 79.4%, yang memungkinkan pengendalian hama secara real-time untuk 
        mendukung praktik pertanian berkelanjutan.
        • Emotion Classification (Agustus 2024): Yudi mengembangkan model klasifikasi 
        emosi menggunakan dataset FastJobs/Visual Emotional Analysis. Dengan melakukan 
        fine-tuning pada model Google-ViT, ia mencapai akurasi 63.75% dengan loss 1.0965. 
        Model ini telah diunduh sebanyak 237 kali, menunjukkan nilai dan efektivitasnya untuk 
        pengembangan lebih lanjut.
        • Gesture-Controlled Robot (Agustus 2024): Proyek ini melibatkan pengembangan 
        robotic hand yang dikendalikan melalui gerakan tangan dengan bantuan Arduino dan 
        mekanisme servo-thread. Penggunaan OpenCV dan Mediapipe memungkinkan 
        pendeteksian real-time terhadap sudut pergerakan jari, yang selanjutnya digunakan 
        untuk mengontrol pergerakan motor servo secara presisi.

        - Jawab pertanyaan dengan detail, harus sangat detail, namun jika pertanyaannya simpel dan spesifik jawab saja dengan simpel.
        - Jika pertanyaannya tidak jelas, asumsikan pertanyaan tersebut mengacu pada latar belakang profesional, keterampilan, atau pengalamannya.
        - Jika memang anda tidak tahu mengenai jawabannya jangan mengarang jawaban, jawab saja mohon maaf saya tidak tahu.
        - Jika pertanyaan nya tidak jelas atau anda tidak mengerti pertanyanyaannya, jawab saja anda tidak mengerti jawabannya, jangan anda paksakan untuk menjawab terutama jika tidak ada di konteks. jika pertanyaannya tidak jelas anda tidak perlu menjawab berdasarkan history juga. Oleh karena itu, anda harus bilang tidak mengerti dengan pertanyaannya.

        Pertanyaan: {question}  
        Konteks: {context}
        History: {history}
        [/INST]
    """

    template = template_en if language == "English" else template_id
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    return chain.stream({"question": question, "context": context, "history": history})

index_id = FAISS.load_local("faiss-index/faiss_index_indonesia", embeddings=NomicEmbeddings(
    model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY), allow_dangerous_deserialization=True)
index_en = FAISS.load_local("faiss-index/faiss_index_indonesia", embeddings=NomicEmbeddings(
    model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY), allow_dangerous_deserialization=True)

index = index_en if language == "English" else index_id

with st.chat_message("assistant"):
    st.write("Yudi: ")
    st.markdown(st.session_state.intro[language])

for chat in st.session_state.message:
    if isinstance(chat, HumanMessage):
        with st.chat_message("User"):
            st.write("User: ")
            st.markdown(chat.content)
    else:
        with st.chat_message("assistant"):
            st.write("Yudi: ")
            st.markdown(chat.content)

question = st.chat_input("Enter Your Question")

if question != "" and question is not None:
    st.session_state.message.append(HumanMessage(question))
    with st.chat_message("User"):
        st.write("User: ")
        st.markdown(question)
    with st.spinner("Generating response..."):
        related_documents = retrieve_docs(question, index)
        # answer = answer_question(question, related_documents)

    with st.chat_message("assistant"):
        st.write("Yudi: ")
        try:
            answer = st.write_stream(answer_question(question, related_documents, st.session_state.message))
            st.session_state.message.append(AIMessage(answer))
        except groq.APIStatusError as e:
            if "rate_limit_exceeded" in str(e):
                st.session_state.message = []  # Reset chat history
                st.warning("Chat history terlalu panjang, sehingga telah direset. Silakan coba lagi.")
            else:
                st.error(f"Terjadi error: {e}")