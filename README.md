# 📌 LangChain FAISS & Groq Chatbot

Proyek ini adalah chatbot berbasis AI yang menggunakan **LangChain**, **FAISS**, dan **Groq** untuk membangun sistem tanya jawab berbasis Retrieval-Augmented Generation (RAG). Aplikasi ini dibuat dengan **Streamlit** sebagai antarmuka pengguna.
Proyek ini juga mendukung penggunaan dokumen pribadi dengan embedding menggunakan Nomic.
🔗 Coba chatbot sekarang: [Yudi Chatbot](https://yudichatbot.streamlit.app/)

## 🚀 Fitur Utama
- **Vector Store dengan FAISS** untuk penyimpanan dan pencarian embedding dokumen.
- **Prompt Engineering** dengan `ChatPromptTemplate` dari LangChain.
- **Model AI Groq** untuk menghasilkan jawaban dari input pengguna.
- **Integrasi Nomic Embeddings** untuk representasi teks ke dalam vektor.
- **Antarmuka Streamlit** untuk memudahkan interaksi dengan chatbot.

## 📂 Struktur Proyek
```
📂 project-root
│── chatbot.py  # File utama Streamlit
│── vector_embed.py  # File melakukan embedding dan menyimpan vector ke faiss
│── requirements.txt  # Dependensi proyek
│── README.md  # Dokumentasi proyek
└── data/  # (Opsional) Folder untuk menyimpan dokumen referensi
```

## 🛠 Instalasi dan Menjalankan Proyek
Pastikan Anda sudah menginstal Python 3.8 atau lebih baru.

1. Clone repository ini:
   ```sh
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```
2. Buat dan aktifkan virtual environment (opsional tetapi disarankan):
   ```sh
   python -m venv venv
   source venv/bin/activate  # Untuk macOS/Linux
   venv\Scripts\activate     # Untuk Windows
   ```
3. Instal dependensi:
   ```sh
   pip install -r requirements.txt
   ```

## Penggunaan
### Menggunakan Dokumen Pribadi
1. **Dapatkan API Key Nomic:**
   - Pergi ke [Nomic Atlas](https://atlas.nomic.ai/) dan buat akun jika belum memiliki.
   - Dapatkan API key dari dashboard.
   
2. **Simpan API Key dalam file `.env`**:
   Buat file `.env` dan tambahkan API key:
   ```env
   NOMIC_API_KEY=your_nomic_api_key
   ```

3. **Gunakan dotenv untuk memuat API key**:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   api_key = os.getenv("NOMIC_API_KEY")
   ```

4. **Jalankan `vector_embed.py` untuk membuat embedding**:
   ```sh
   python vector_embed.py
   ```

5. **Dapatkan API Key Groq:**
   - Pergi ke [Groq Console](https://console.groq.com/keys)
   - Buat dan salin API key
   - Tambahkan ke `.env`:
     ```env
     GROQ_API_KEY=your_groq_api_key
     ```

6. **Lakukan hal yang sama seperti langkah 3 untuk memuat API key di `chatbot.py`.**

7. **Jalankan chatbot:**
   ```sh
   streamlit run chatbot.py
   ```

## Keamanan API Key
Jangan pernah mengunggah file `.env` ke GitHub! Tambahkan `.env` ke `.gitignore` untuk mencegah kebocoran API key:
```sh
echo ".env" >> .gitignore
```

## 📌 Teknologi yang Digunakan
- **[LangChain](https://www.langchain.com/)** - Framework untuk membangun aplikasi berbasis LLM.
- **[FAISS](https://faiss.ai/)** - Library untuk pencarian berbasis vektor.
- **[Groq API](https://groq.com/)** - Model AI untuk chatbot.
- **[Streamlit](https://streamlit.io/)** - Framework untuk membangun UI berbasis Python.

## 💡 Kontribusi
Kontribusi selalu terbuka! Silakan buat **pull request** atau buka **issue** jika menemukan bug atau ingin menambahkan fitur baru.

---
Dibuat dengan ❤️ oleh [Yudi Pratama Putra](https://github.com/Yud1Pp)

---

# 📌 LangChain FAISS & Groq Chatbot (English Version)

This project is an AI-based chatbot that utilizes **LangChain**, **FAISS**, and **Groq** to build a Retrieval-Augmented Generation (RAG)-based question-answering system. The application is built using **Streamlit** as the user interface.
This project also supports personal document usage with embeddings using Nomic.
🔗 Try the chatbot now: [Yudi Chatbot](https://yudichatbot.streamlit.app/)

## 🚀 Main Features
- **Vector Store with FAISS** for document embedding storage and retrieval.
- **Prompt Engineering** with `ChatPromptTemplate` from LangChain.
- **Groq AI Model** to generate answers from user inputs.
- **Nomic Embeddings Integration** for text-to-vector representation.
- **Streamlit Interface** for easy chatbot interaction.

## 📂 Project Structure
```
📂 project-root
│── chatbot.py  # Main Streamlit file
│── vector_embed.py  # Handles embedding and storing vectors in FAISS
│── requirements.txt  # Project dependencies
│── README.md  # Project documentation
└── data/  # (Optional) Folder to store reference documents
```

## 🛠 Installation and Running the Project
Make sure you have Python 3.8 or newer installed.

1. Clone this repository:
   ```sh
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate     # For Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Using Personal Documents
1. **Get Nomic API Key:**
   - Go to [Nomic Atlas](https://atlas.nomic.ai/) and create an account if you don’t have one.
   - Retrieve your API key from the dashboard.
   
2. **Store API Key in `.env` file**:
   Create a `.env` file and add the API key:
   ```env
   NOMIC_API_KEY=your_nomic_api_key
   ```

3. **Use dotenv to load API key**:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   api_key = os.getenv("NOMIC_API_KEY")
   ```

4. **Run `vector_embed.py` to create embeddings**:
   ```sh
   python vector_embed.py
   ```

5. **Get Groq API Key:**
   - Visit [Groq Console](https://console.groq.com/keys)
   - Generate and copy the API key
   - Add it to `.env`:
     ```env
     GROQ_API_KEY=your_groq_api_key
     ```

6. **Repeat step 3 to load API key in `chatbot.py`.**

7. **Run the chatbot:**
   ```sh
   streamlit run chatbot.py
   ```

## API Key Security
Never upload the `.env` file to GitHub! Add `.env` to `.gitignore` to prevent API key leaks:
```sh
echo ".env" >> .gitignore
```

---
Created with ❤️ by [Yudi Pratama Putra](https://github.com/Yud1Pp)
