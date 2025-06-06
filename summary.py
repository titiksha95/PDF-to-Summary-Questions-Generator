import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QTextEdit, QLabel, QFileDialog, 
                            QSpinBox, QGroupBox, QMessageBox, QLineEdit, QSplitter,
                            QTabWidget, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import PyPDF2
import fitz  # PyMuPDF - alternative PDF reader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import random

class PDFProcessor(QThread):
    """Background thread for processing PDF operations"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, operation, pdf_path, pages=None, question=None, question_types=None):
        super().__init__()
        self.operation = operation
        self.pdf_path = pdf_path
        self.pages = pages
        self.question = question
        self.question_types = question_types or []
        self.summarizer = None
        self.qa_pipeline = None
        
    def run(self):
        try:
            if self.operation == "summarize":
                result = self.summarize_pdf()
            elif self.operation == "question":
                result = self.answer_question()
            elif self.operation == "generate_questions":
                result = self.generate_questions()
            elif self.operation == "test":
                result = self.test_pdf()
            else:
                result = "Invalid operation"
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def extract_text_from_pdf(self):
        """Extract text from PDF using PyMuPDF for better text extraction"""
        text = ""
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            
            if self.pages:
                pages_to_process = [p-1 for p in self.pages if 0 <= p-1 < total_pages]
            else:
                pages_to_process = list(range(total_pages))
            
            for i, page_num in enumerate(pages_to_process):
                page = doc[page_num]
                text += page.get_text()
                self.progress.emit(int((i + 1) / len(pages_to_process) * 30))
            
            doc.close()
            
        except Exception as e:
            # Fallback to PyPDF2
            try:
                with open(self.pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    if self.pages:
                        pages_to_process = [p-1 for p in self.pages if 0 <= p-1 < total_pages]
                    else:
                        pages_to_process = list(range(total_pages))
                    
                    for i, page_num in enumerate(pages_to_process):
                        page = reader.pages[page_num]
                        text += page.extract_text()
                        self.progress.emit(int((i + 1) / len(pages_to_process) * 30))
                        
            except Exception as e2:
                raise Exception(f"Failed to extract text: {str(e2)}")
        
        return text.strip()
    
    def test_pdf(self):
        """Test PDF file reading capabilities"""
        test_results = []
        test_results.append("🔧 PDF File Test Results")
        test_results.append("=" * 50)
        test_results.append(f"📁 File: {os.path.basename(self.pdf_path)}")
        test_results.append(f"📍 Path: {self.pdf_path}")
        test_results.append(f"💾 Size: {os.path.getsize(self.pdf_path) / 1024:.1f} KB")
        test_results.append("")
        
        # Test PyMuPDF
        try:
            doc = fitz.open(self.pdf_path)
            page_count = len(doc)
            test_results.append("✅ PyMuPDF (fitz) - SUCCESS")
            test_results.append(f"   📄 Pages detected: {page_count}")
            
            # Test first page extraction
            if page_count > 0:
                first_page_text = doc[0].get_text()
                test_results.append(f"   📝 First page text length: {len(first_page_text)} characters")
                if first_page_text.strip():
                    test_results.append("   ✅ Text extraction working")
                    # Show first 200 characters as preview
                    preview = first_page_text.strip()[:200] + "..." if len(first_page_text) > 200 else first_page_text.strip()
                    test_results.append(f"   📖 Preview: {preview}")
                else:
                    test_results.append("   ⚠️ No text found on first page")
            doc.close()
        except Exception as e:
            test_results.append(f"❌ PyMuPDF (fitz) - FAILED: {str(e)}")
        
        test_results.append("")
        
        # Test PyPDF2
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                test_results.append("✅ PyPDF2 - SUCCESS")
                test_results.append(f"   📄 Pages detected: {page_count}")
                
                # Test first page extraction
                if page_count > 0:
                    first_page_text = reader.pages[0].extract_text()
                    test_results.append(f"   📝 First page text length: {len(first_page_text)} characters")
                    if first_page_text.strip():
                        test_results.append("   ✅ Text extraction working")
                    else:
                        test_results.append("   ⚠️ No text found on first page")
        except Exception as e:
            test_results.append(f"❌ PyPDF2 - FAILED: {str(e)}")
        
        test_results.append("")
        test_results.append("🧠 AI Model Availability:")
        
        # Test model availability
        try:
            # Test summarization model
            pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
            test_results.append("   ✅ Summarization model available")
        except Exception as e:
            test_results.append(f"   ❌ Summarization model failed: {str(e)}")
        
        try:
            # Test QA model
            pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
            test_results.append("   ✅ Question-answering model available")
        except Exception as e:
            test_results.append(f"   ❌ Question-answering model failed: {str(e)}")
        
        test_results.append("")
        test_results.append("📊 Recommendations:")
        test_results.append("   • If both PDF readers work, PyMuPDF usually provides better text extraction")
        test_results.append("   • If no text is extracted, the PDF might be image-based (scanned)")
        test_results.append("   • First-time model loading may take a few minutes")
        
        return "\n".join(test_results)
    
    def generate_questions(self):
        """Generate possible questions from PDF content"""
        text = self.extract_text_from_pdf()
        
        if not text:
            return "No text found in the specified pages."
        
        self.progress.emit(40)
        
        # Clean and prepare text
        text = self.clean_text(text)
        
        questions = []
        
        # Generate different types of questions based on selected types
        if "factual" in self.question_types:
            questions.extend(self.generate_factual_questions(text))
            self.progress.emit(55)
        
        if "conceptual" in self.question_types:
            questions.extend(self.generate_conceptual_questions(text))
            self.progress.emit(70)
        
        if "analytical" in self.question_types:
            questions.extend(self.generate_analytical_questions(text))
            self.progress.emit(85)
        
        if "application" in self.question_types:
            questions.extend(self.generate_application_questions(text))
            self.progress.emit(95)
        
        # If no specific types selected, generate all types
        if not self.question_types:
            questions.extend(self.generate_factual_questions(text))
            questions.extend(self.generate_conceptual_questions(text))
            questions.extend(self.generate_analytical_questions(text))
            questions.extend(self.generate_application_questions(text))
        
        self.progress.emit(100)
        
        if questions:
            # Remove duplicates and format
            unique_questions = list(set(questions))
            random.shuffle(unique_questions)  # Mix up the order
            
            result = "🤔 GENERATED QUESTIONS FROM PDF CONTENT\n"
            result += "=" * 60 + "\n\n"
            
            # Group questions by type
            factual_q = [q for q in unique_questions if any(starter in q.lower() for starter in ['what is', 'who is', 'when', 'where', 'which'])]
            conceptual_q = [q for q in unique_questions if any(starter in q.lower() for starter in ['explain', 'describe', 'define', 'how does'])]
            analytical_q = [q for q in unique_questions if any(starter in q.lower() for starter in ['why', 'analyze', 'compare', 'evaluate'])]
            application_q = [q for q in unique_questions if any(starter in q.lower() for starter in ['how can', 'apply', 'implement', 'use'])]
            
            if factual_q and ("factual" in self.question_types or not self.question_types):
                result += "📋 FACTUAL QUESTIONS:\n" + "-" * 25 + "\n"
                for i, q in enumerate(factual_q[:8], 1):
                    result += f"{i}. {q}\n"
                result += "\n"
            
            if conceptual_q and ("conceptual" in self.question_types or not self.question_types):
                result += "💡 CONCEPTUAL QUESTIONS:\n" + "-" * 30 + "\n"
                for i, q in enumerate(conceptual_q[:8], 1):
                    result += f"{i}. {q}\n"
                result += "\n"
            
            if analytical_q and ("analytical" in self.question_types or not self.question_types):
                result += "🔍 ANALYTICAL QUESTIONS:\n" + "-" * 28 + "\n"
                for i, q in enumerate(analytical_q[:6], 1):
                    result += f"{i}. {q}\n"
                result += "\n"
            
            if application_q and ("application" in self.question_types or not self.question_types):
                result += "⚙️ APPLICATION QUESTIONS:\n" + "-" * 29 + "\n"
                for i, q in enumerate(application_q[:6], 1):
                    result += f"{i}. {q}\n"
                result += "\n"
            
            result += f"📊 TOTAL QUESTIONS GENERATED: {len(unique_questions)}\n"
            result += "💡 TIP: Click any question type above to focus your study!"
            
            return result
        else:
            return "Could not generate questions from the extracted text. The content might be too short or unclear."
    
    def clean_text(self, text):
        """Clean and prepare text for question generation"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text.strip()
    
    def generate_factual_questions(self, text):
        """Generate factual questions (What, Who, When, Where, Which)"""
        questions = []
        
        # Extract key terms and concepts
        sentences = text.split('.')
        
        # Look for definitions and key terms
        for sentence in sentences[:20]:  # Limit to avoid too many questions
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Look for patterns like "X is Y" or "X are Y"
                if ' is ' in sentence.lower():
                    parts = sentence.split(' is ')
                    if len(parts) >= 2 and len(parts[0].split()) <= 4:
                        questions.append(f"What is {parts[0].strip()}?")
                
                # Look for numbers and dates
                if re.search(r'\d{4}', sentence):  # Years
                    questions.append(f"When was {sentence.split()[0]} established/created?")
                
                # Look for lists and enumerations
                if 'include' in sentence.lower() or 'such as' in sentence.lower():
                    questions.append(f"What are the main components mentioned in this section?")
        
        # Add generic factual questions
        questions.extend([
            "What are the main topics covered in this document?",
            "What key terms are defined in this section?",
            "Which concepts are most frequently mentioned?",
            "What examples are provided in the text?"
        ])
        
        return questions[:10]  # Limit number of questions
    
    def generate_conceptual_questions(self, text):
        """Generate conceptual questions (Explain, Describe, Define, How does)"""
        questions = []
        
        # Look for complex concepts
        key_terms = self.extract_key_terms(text)
        
        for term in key_terms[:8]:
            questions.extend([
                f"Explain the concept of {term}.",
                f"How does {term} work?",
                f"Describe the main features of {term}."
            ])
        
        # Add generic conceptual questions
        questions.extend([
            "Explain the main concepts discussed in this section.",
            "How do the different topics relate to each other?",
            "Describe the overall structure of the content.",
            "Explain the significance of the information presented."
        ])
        
        return questions[:12]
    
    def generate_analytical_questions(self, text):
        """Generate analytical questions (Why, Analyze, Compare, Evaluate)"""
        questions = [
            "Why is this topic important?",
            "Analyze the main arguments presented in the text.",
            "Compare and contrast the different approaches mentioned.",
            "Evaluate the strengths and weaknesses of the concepts discussed.",
            "Why might someone choose one method over another?",
            "Analyze the relationship between the key concepts.",
            "What are the advantages and disadvantages mentioned?",
            "Why do these concepts matter in practical applications?"
        ]
        
        return questions
    
    def generate_application_questions(self, text):
        """Generate application questions (How can, Apply, Implement, Use)"""
        questions = [
            "How can you apply these concepts in real-world scenarios?",
            "What are practical ways to implement the ideas discussed?",
            "How would you use this information in your work/studies?",
            "What steps would you take to apply these principles?",
            "How can the concepts be adapted for different situations?",
            "What tools or methods could help implement these ideas?",
            "How would you explain these concepts to someone else?",
            "What practical exercises could reinforce this learning?"
        ]
        
        return questions
    
    def extract_key_terms(self, text):
        """Extract key terms from text"""
        # Simple key term extraction based on frequency and capitalization
        words = text.split()
        
        # Look for capitalized words (potential key terms)
        key_terms = []
        for word in words:
            word = re.sub(r'[^\w]', '', word)  # Remove punctuation
            if (len(word) > 3 and 
                word[0].isupper() and 
                word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']):
                key_terms.append(word)
        
        # Return most frequent key terms
        from collections import Counter
        term_freq = Counter(key_terms)
        return [term for term, freq in term_freq.most_common(10)]
    
    def summarize_pdf(self):
        """Summarize the PDF content"""
        text = self.extract_text_from_pdf()
        
        if not text:
            return "No text found in the specified pages."
        
        self.progress.emit(60)
        
        # Initialize summarizer if not already done
        if not self.summarizer:
            try:
                # Use a lightweight model for summarization
                model_name = "facebook/bart-large-cnn"
                self.summarizer = pipeline("summarization", 
                                         model=model_name,
                                         device=0 if torch.cuda.is_available() else -1)
            except Exception:
                # Fallback to a smaller model
                try:
                    model_name = "sshleifer/distilbart-cnn-12-6"
                    self.summarizer = pipeline("summarization", 
                                             model=model_name,
                                             device=-1)  # Use CPU
                except Exception:
                    return "Error: Could not load summarization model. Please install required models."
        
        self.progress.emit(80)
        
        # Split text into chunks if it's too long
        max_chunk_length = 1024
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                try:
                    summary = self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    summaries.append(f"Error summarizing chunk {i+1}: {str(e)}")
            
            self.progress.emit(80 + int((i + 1) / len(chunks) * 20))
        
        if summaries:
            return "\n\n".join(summaries)
        else:
            return "Could not generate summary from the extracted text."
    
    def answer_question(self):
        """Answer question based on PDF content"""
        text = self.extract_text_from_pdf()
        
        if not text:
            return "No text found in the specified pages to answer the question."
        
        self.progress.emit(60)
        
        # Initialize QA pipeline if not already done
        if not self.qa_pipeline:
            try:
                # Use a lightweight QA model
                model_name = "distilbert-base-cased-distilled-squad"
                self.qa_pipeline = pipeline("question-answering", 
                                          model=model_name,
                                          device=0 if torch.cuda.is_available() else -1)
            except Exception:
                try:
                    # Fallback model
                    model_name = "deepset/minilm-uncased-squad2"
                    self.qa_pipeline = pipeline("question-answering", 
                                              model=model_name,
                                              device=-1)
                except Exception:
                    return "Error: Could not load question-answering model. Please install required models."
        
        self.progress.emit(80)
        
        # Limit context length for the model
        max_context_length = 4000
        if len(text) > max_context_length:
            text = text[:max_context_length]
        
        try:
            result = self.qa_pipeline(question=self.question, context=text)
            confidence = result['score']
            answer = result['answer']
            
            response = f"Answer: {answer}\n\nConfidence: {confidence:.2%}"
            
            if confidence < 0.3:
                response += "\n\nNote: Low confidence answer. The information might not be directly available in the text."
            
            return response
        
        except Exception as e:
            return f"Error answering question: {str(e)}"

class PDFToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pdf_path = None
        self.processor_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("PDF Summarizer & Q&A Tool with Question Generation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: Arial;
                font-size: 12px;
            }
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }
            QCheckBox {
                font-size: 12px;
                padding: 2px;
            }
        """)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # File selection group
        file_group = QGroupBox("PDF File Selection")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        self.browse_btn = QPushButton("Browse PDF")
        self.browse_btn.clicked.connect(self.browse_file)
        
        # Auto-load button for the specific PDF
        self.auto_load_btn = QPushButton("Default")
        self.auto_load_btn.clicked.connect(self.auto_load_python_guide)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.auto_load_btn)
        
        # Page selection group
        page_group = QGroupBox("Page Selection")
        page_layout = QHBoxLayout(page_group)
        
        page_layout.addWidget(QLabel("Start Page:"))
        self.start_page = QSpinBox()
        self.start_page.setMinimum(1)
        self.start_page.setMaximum(9999)
        self.start_page.setValue(1)
        
        page_layout.addWidget(self.start_page)
        
        page_layout.addWidget(QLabel("End Page:"))
        self.end_page = QSpinBox()
        self.end_page.setMinimum(1)
        self.end_page.setMaximum(9999)
        self.end_page.setValue(1)
        
        page_layout.addWidget(self.end_page)
        
        self.all_pages_btn = QPushButton("All Pages")
        self.all_pages_btn.clicked.connect(self.select_all_pages)
        page_layout.addWidget(self.all_pages_btn)
        
        page_layout.addStretch()
        
        # Question Generation Options
        question_gen_group = QGroupBox("Question Generation Options")
        question_gen_layout = QVBoxLayout(question_gen_group)
        
        # Checkboxes for question types
        checkbox_layout = QHBoxLayout()
        self.factual_cb = QCheckBox("📋 Factual Questions")
        self.conceptual_cb = QCheckBox("💡 Conceptual Questions")
        self.analytical_cb = QCheckBox("🔍 Analytical Questions")
        self.application_cb = QCheckBox("⚙️ Application Questions")
        
        # Set all checked by default
        self.factual_cb.setChecked(True)
        self.conceptual_cb.setChecked(True)
        self.analytical_cb.setChecked(True)
        self.application_cb.setChecked(True)
        
        checkbox_layout.addWidget(self.factual_cb)
        checkbox_layout.addWidget(self.conceptual_cb)
        checkbox_layout.addWidget(self.analytical_cb)
        checkbox_layout.addWidget(self.application_cb)
        checkbox_layout.addStretch()
        
        question_gen_layout.addLayout(checkbox_layout)
        
        # Operations group
        operations_group = QGroupBox("Operations")
        operations_layout = QHBoxLayout(operations_group)
        
        self.summarize_btn = QPushButton("📄 Summarize PDF")
        self.summarize_btn.clicked.connect(self.summarize_pdf)
        self.summarize_btn.setEnabled(False)
        
        self.question_btn = QPushButton("❓ Ask Question")
        self.question_btn.clicked.connect(self.ask_question)
        self.question_btn.setEnabled(False)
        
        # NEW: Generate Questions button
        self.generate_questions_btn = QPushButton("🤔 Generate Questions")
        self.generate_questions_btn.clicked.connect(self.generate_questions)
        self.generate_questions_btn.setEnabled(False)
        self.generate_questions_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        
        # Quick test button
        self.test_btn = QPushButton("🧪 Test PDF")
        self.test_btn.clicked.connect(self.test_pdf_file)
        
        operations_layout.addWidget(self.summarize_btn)
        operations_layout.addWidget(self.question_btn)
        operations_layout.addWidget(self.generate_questions_btn)
        operations_layout.addWidget(self.test_btn)
        operations_layout.addStretch()
        
        # Question input
        question_group = QGroupBox("Question Input")
        question_layout = QVBoxLayout(question_group)
        
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your question about the PDF content...")
        self.question_input.returnPressed.connect(self.ask_question)
        
        # Quick question buttons
        quick_questions_layout = QHBoxLayout()
        quick_q1 = QPushButton("What are the main topics?")
        quick_q2 = QPushButton("Key concepts?")
        quick_q3 = QPushButton("Important features?")
        
        quick_q1.clicked.connect(lambda: self.set_question("What are the main topics covered in this document?"))
        quick_q2.clicked.connect(lambda: self.set_question("What are the key Python concepts explained?"))
        quick_q3.clicked.connect(lambda: self.set_question("What are the most important Python features mentioned?"))
        
        quick_questions_layout.addWidget(quick_q1)
        quick_questions_layout.addWidget(quick_q2)
        quick_questions_layout.addWidget(quick_q3)
        quick_questions_layout.addStretch()
        
        question_layout.addWidget(self.question_input)
        question_layout.addLayout(quick_questions_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Results area with tabs
        self.tab_widget = QTabWidget()
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setPlaceholderText("PDF summary will appear here...")
        self.tab_widget.addTab(self.summary_text, "Summary")
        
        # Q&A tab
        self.qa_text = QTextEdit()
        self.qa_text.setPlaceholderText("Question answers will appear here...")
        self.tab_widget.addTab(self.qa_text, "Q&A Results")
        
        # NEW: Generated Questions tab
        self.generated_questions_text = QTextEdit()
        self.generated_questions_text.setPlaceholderText("Generated questions will appear here...")
        self.tab_widget.addTab(self.generated_questions_text, "📝 Generated Questions")
        
        # Test results tab
        self.test_text = QTextEdit()
        self.test_text.setPlaceholderText("PDF test results will appear here...")
        self.tab_widget.addTab(self.test_text, "Test Results")
        
        # Add all components to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(page_group)
        main_layout.addWidget(question_gen_group)
        main_layout.addWidget(operations_group)
        main_layout.addWidget(question_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.tab_widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Please select a PDF file or click 'Default'")
        
    def auto_load_python_guide(self):
        """Auto-load the specific Python Theory PDF"""
        pdf_path = r"C:\Users\Vicky Singh\Downloads\Python_Theory_Only_Guide.pdf"
        
        if os.path.exists(pdf_path):
            self.load_pdf_file(pdf_path)
        else:
            QMessageBox.warning(self, "File Not Found", 
                              f"Could not find the Python Theory Guide at:\n{pdf_path}\n\n"
                              "Please use 'Browse PDF' to select the correct file.")
    
    def set_question(self, question):
        """Set a predefined question"""
        self.question_input.setText(question)
        
    def browse_file(self):
        """Browse and select PDF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self.load_pdf_file(file_path)
    
    def load_pdf_file(self, file_path):
        """Load a PDF file and update UI"""
        self.pdf_path = file_path
        filename = os.path.basename(file_path)
        self.file_label.setText(f"Selected: {filename}")
        self.file_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        
        # Enable operation buttons
        self.summarize_btn.setEnabled(True)
        self.question_btn.setEnabled(True)
        self.generate_questions_btn.setEnabled(True)
        
        # Try to get page count
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                self.end_page.setMaximum(page_count)
                self.end_page.setValue(page_count)
                self.start_page.setMaximum(page_count)
                self.statusBar().showMessage(f"PDF loaded - {page_count} pages found")
        except Exception as e:
            self.statusBar().showMessage("PDF loaded - Could not determine page count")
    
    def select_all_pages(self):
        """Select all pages"""
        self.start_page.setValue(1)
        self.end_page.setValue(self.end_page.maximum())
    
    def get_selected_pages(self):
        """Get list of selected pages"""
        start = self.start_page.value()
        end = self.end_page.value()
        
        if start > end:
            start, end = end, start
            
        return list(range(start, end + 1))
    
    def get_selected_question_types(self):
        """Get selected question types"""
        types = []
        if self.factual_cb.isChecked():
            types.append("factual")
        if self.conceptual_cb.isChecked():
            types.append("conceptual")
        if self.analytical_cb.isChecked():
            types.append("analytical")
        if self.application_cb.isChecked():
            types.append("application")
        return types
    
    def generate_questions(self):
        """Generate questions from PDF content"""
        if not self.pdf_path:
            QMessageBox.warning(self, "Warning", "Please select a PDF file first.")
            return
        
        question_types = self.get_selected_question_types()
        if not question_types:
            QMessageBox.warning(self, "Warning", "Please select at least one question type.")
            return
        
        pages = self.get_selected_pages()
        self.start_processing("generate_questions", pages, question_types=question_types)
    
    def test_pdf_file(self):
        """Test PDF file reading capabilities"""
        if not self.pdf_path:
            QMessageBox.warning(self, "Warning", "Please select a PDF file first.")
            return
        
        self.start_processing("test")
    
    def summarize_pdf(self):
        """Start PDF summarization"""
        if not self.pdf_path:
            QMessageBox.warning(self, "Warning", "Please select a PDF file first.")
            return
        
        pages = self.get_selected_pages()
        self.start_processing("summarize", pages)
    
    def ask_question(self):
        """Ask question about PDF content"""
        if not self.pdf_path:
            QMessageBox.warning(self, "Warning", "Please select a PDF file first.")
            return
        
        question = self.question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
            return
        
        pages = self.get_selected_pages()
        self.start_processing("question", pages, question)
    
    def start_processing(self, operation, pages=None, question=None, question_types=None):
        """Start background processing"""
        if self.processor_thread and self.processor_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Another operation is already running. Please wait.")
            return
        
        # Disable buttons and show progress
        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Update status
        operation_names = {
            "summarize": "Summarizing PDF...",
            "question": "Answering question...",
            "generate_questions": "Generating questions...",
            "test": "Testing PDF file..."
        }
        self.statusBar().showMessage(operation_names.get(operation, "Processing..."))
        
        # Start processing thread
        self.processor_thread = PDFProcessor(operation, self.pdf_path, pages, question, question_types)
        self.processor_thread.finished.connect(self.on_processing_finished)
        self.processor_thread.error.connect(self.on_processing_error)
        self.processor_thread.progress.connect(self.progress_bar.setValue)
        self.processor_thread.start()
    
    def on_processing_finished(self, result):
        """Handle processing completion"""
        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Processing completed successfully")
        
        # Determine which tab to show and update
        if self.processor_thread.operation == "summarize":
            self.summary_text.setPlainText(result)
            self.tab_widget.setCurrentIndex(0)  # Summary tab
        elif self.processor_thread.operation == "question":
            # Add to Q&A history
            question = self.processor_thread.question
            current_content = self.qa_text.toPlainText()
            if current_content:
                new_content = current_content + "\n\n" + "="*50 + "\n\n"
            else:
                new_content = ""
            new_content += f"Q: {question}\n\n{result}"
            self.qa_text.setPlainText(new_content)
            self.tab_widget.setCurrentIndex(1)  # Q&A tab
        elif self.processor_thread.operation == "generate_questions":
            self.generated_questions_text.setPlainText(result)
            self.tab_widget.setCurrentIndex(2)  # Generated Questions tab
        elif self.processor_thread.operation == "test":
            self.test_text.setPlainText(result)
            self.tab_widget.setCurrentIndex(3)  # Test Results tab
    
    def on_processing_error(self, error_message):
        """Handle processing error"""
        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Processing failed")
        
        QMessageBox.critical(self, "Processing Error", 
                           f"An error occurred during processing:\n\n{error_message}")
    
    def set_buttons_enabled(self, enabled):
        """Enable/disable operation buttons"""
        self.summarize_btn.setEnabled(enabled and self.pdf_path is not None)
        self.question_btn.setEnabled(enabled and self.pdf_path is not None)
        self.generate_questions_btn.setEnabled(enabled and self.pdf_path is not None)
        self.test_btn.setEnabled(enabled and self.pdf_path is not None)
        self.browse_btn.setEnabled(enabled)
        self.auto_load_btn.setEnabled(enabled)
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.processor_thread and self.processor_thread.isRunning():
            reply = QMessageBox.question(self, "Close Application", 
                                       "Processing is still running. Do you want to close anyway?",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.processor_thread.terminate()
                self.processor_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("PDF Summarizer & Q&A Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("PDF Tools")
    
    # Create and show main window
    window = PDFToolGUI()
    window.show()
    
    # Check for required dependencies
    try:
        import transformers
        import torch
        import fitz
        import PyPDF2
    except ImportError as e:
        QMessageBox.warning(window, "Missing Dependencies", 
                          f"Some required libraries are missing:\n{str(e)}\n\n"
                          "Please install them using:\n"
                          "pip install transformers torch PyMuPDF PyPDF2")
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()