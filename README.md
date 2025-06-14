📝 PDF-to-Summary/Questions Generator 
📌 Overview
This is a desktop application that allows users to:

📄 Summarize the contents of a PDF file

❓ Ask natural language questions about the content

🤔 Automatically generate questions from the PDF

🧪 Test the PDF's readability and model availability

It combines powerful NLP models with an intuitive GUI to assist in study, review, and comprehension of PDF documents.

🛠️ Features
PDF Summarization:

Extracts text from selected pages

Summarizes using HuggingFace models like facebook/bart-large-cnn or distilbart-cnn-12-6

Question Answering:

Uses models like distilbert-base-cased-distilled-squad to find answers in the PDF content

Accepts typed questions and shows confidence score

Auto Question Generation:

Generates questions in four categories:

📋 Factual

💡 Conceptual

🔍 Analytical

⚙️ Application

Uses simple heuristics and extracted key terms for intelligent question design

PDF Testing Utility:

Verifies readability with both PyMuPDF and PyPDF2

Tests if NLP models are available

Shows a preview of extracted text

📂 Dependencies
Ensure the following libraries are installed:

bash
Copy
Edit
pip install PyQt5 PyMuPDF PyPDF2 transformers torch
🚀 How to Run
Save the script as pdf_tool.py (or keep original name).

Run the application:

bash
Copy
Edit
python pdf_tool.py
📋 How to Use
Launch the app.

Click Browse PDF or Default to load a file.

Select page range or choose All Pages.

Choose desired action:

📄 Summarize PDF – To get a summary

❓ Ask Question – Type your question, then click

🤔 Generate Questions – Auto generate study questions

🧪 Test PDF – Ensure everything works

View results in the corresponding tab at the bottom.

📌 Notes
Uses multithreading (QThread) for smooth operation.

Uses PyMuPDF primarily for better text extraction; falls back to PyPDF2.

Includes styling, error handling, and basic input validation.

PDF page range is 1-indexed (like humans count, not like Python lists).

👨‍💻 Developer Notes
Modular design with PDFProcessor handling background tasks

NLP models are lazy-loaded for efficiency

Includes fallback models for both summarization and QA

GUI built with QMainWindow, QVBoxLayout, and QTabWidget for clean structure
