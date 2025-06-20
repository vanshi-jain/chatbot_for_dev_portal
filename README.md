# OCR-to-Chat QA System

This project uses PaddleOCR to extract text from images and LangChain + OpenAI to generate intelligent question-answering capabilities over the extracted text. It also includes evaluation using BLEU, METEOR, and ROUGE scores.


## 🚀 Features

- OCR using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Context-aware Q&A using LangChain and OpenAI GPT
- Text similarity evaluation using BLEU, METEOR, and ROUGE
- Interactive UI built with Streamlit


## Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Download NLTK resources
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
### 5. Create a .env file
Add your OpenAI API key to a .env file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key
```

## Running the App
Start the Streamlit app:
```
streamlit run streamlit_app.py
```
