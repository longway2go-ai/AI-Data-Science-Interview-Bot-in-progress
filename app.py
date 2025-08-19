# app.py

import os
import io
import json
import random
import re
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from groq import Groq
import openai
import speech_recognition as sr
import pyttsx3
import tempfile
import uvicorn
from PyPDF2 import PdfReader


# ==== Config and Initialization ====
app = FastAPI()
if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ==== Quiz State ====
class QuizState:
    def __init__(self):
        self.topics = []
        self.current_question_index = 0
        self.qa_history = []
        self.total_score = 0
        self.backend = "gemini"
        self.gemini_api_key = ""
        self.groq_api_key = ""
        self.openai_api_key = ""
        self.total_questions = 40
        self.questions_answered = 0
        self.quiz_ended_early = False
        self.current_topic = ""
        self.difficulty = "easy"
        self.current_question_data = None
        self.document_type = "resume"
        self.document_content = ""

    def reset(self):
        self.topics = []
        self.current_question_index = 0
        self.qa_history = []
        self.total_score = 0
        self.total_questions = 40
        self.questions_answered = 0
        self.quiz_ended_early = False
        self.current_topic = ""
        self.difficulty = "easy"
        self.current_question_data = None
        self.document_type = "resume"
        self.document_content = ""


state = QuizState()


# ==== LLM Queries ====
def gemini_query(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")


def groq_query(prompt, api_key):
    try:
        groq_client = Groq(api_key=api_key)
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Groq API Error: {str(e)}")


def openai_query(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OpenAI API Error: {str(e)}")


def llm_query(prompt, backend="gemini"):
    if backend == "gemini":
        api_key = state.gemini_api_key
        if not api_key:
            raise Exception("Gemini API key not configured")
        return gemini_query(prompt, api_key)
    elif backend == "groq":
        api_key = state.groq_api_key
        if not api_key:
            raise Exception("Groq API key not configured")
        return groq_query(prompt, api_key)
    elif backend == "openai":
        api_key = state.openai_api_key
        if not api_key:
            raise Exception("OpenAI API key not configured")
        return openai_query(prompt, api_key)
    else:
        raise Exception(f"Unsupported backend: {backend}")


# ==== File text extraction ====
def extract_text_from_file(file: UploadFile) -> str:
    try:
        if file.content_type == "application/pdf":
            pdf_bytes = file.file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        elif file.content_type.startswith("text/"):
            text = file.file.read().decode("utf-8")
            return text.strip()
        else:
            return ""
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


# ==== Core Functions ====
def extract_document_topics(document_text, document_type, backend):
    """Extract topics from any type of document"""
    
    print(f"=== DEBUG: Starting topic extraction ===")
    print(f"Document type: {document_type}")
    print(f"Backend: {backend}")
    print(f"Document length: {len(document_text)} characters")
    
    # Fallback topics for the three supported document types
    fallback_topics = {
        "resume": ["Technical Skills", "Work Experience", "Education", "Programming Languages", "Frameworks", "Database Management", "Problem Solving", "Project Management"],
        "research_paper": ["Research Methods", "Data Analysis", "Literature Review", "Methodology", "Statistical Analysis", "Hypothesis Testing", "Results Interpretation", "Academic Writing"],
        "technical_document": ["System Architecture", "Implementation", "API Design", "Best Practices", "Configuration", "Troubleshooting", "Documentation", "Technical Specifications"]
    }
    
    document_prompts = {
        "resume": """
        Analyze this resume and extract 8-10 main technical topics, skills, or job roles that would be good for interview questions.
        Focus on technical skills, programming languages, frameworks, tools, and job responsibilities.
        Return only a Python list format like: ['Python', 'Machine Learning', 'API Development', 'Database Design']
        """,
        
        "research_paper": """
        Analyze this research paper and extract 8-10 main topics, concepts, methodologies, or key findings that would be good for quiz questions.
        Focus on key research concepts, methodologies, theoretical frameworks, findings, and technical terms.
        Return only a Python list format like: ['Deep Learning', 'Neural Networks', 'Data Analysis', 'Statistical Methods']
        """,
        
        "technical_document": """
        Analyze this technical document and extract 8-10 main topics, technologies, or processes that would be good for quiz questions.
        Focus on technical concepts, procedures, tools, best practices, and implementation details.
        Return only a Python list format like: ['System Architecture', 'API Design', 'Security Protocols', 'Performance Optimization']
        """
    }
    
    prompt_template = document_prompts.get(document_type, document_prompts["resume"])
    
    prompt = f"""
    {prompt_template}
    
    Document content:
    {document_text[:3000]}...
    
    IMPORTANT: Return ONLY a valid Python list of strings. No explanations or additional text.
    Example format: ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']
    """
    
    try:
        print("=== DEBUG: Attempting AI query ===")
        topics_text = llm_query(prompt, backend)
        print(f"=== DEBUG: Raw AI response: {topics_text[:200]}... ===")
        
        # Try to extract a list from the response
        list_match = re.search(r'\[.*?\]', topics_text, re.DOTALL)
        if list_match:
            try:
                import ast
                list_str = list_match.group(0)
                parsed = ast.literal_eval(list_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid_topics = [str(t).strip() for t in parsed if t and len(str(t).strip()) > 2]
                    if len(valid_topics) >= 3:
                        print(f"=== DEBUG: Successfully extracted {len(valid_topics)} topics ===")
                        return valid_topics[:10]
            except Exception as e:
                print(f"=== DEBUG: Error parsing list: {e} ===")
        
        # Fallback: split by commas and clean up
        if ',' in topics_text:
            topics = []
            for item in topics_text.split(','):
                clean_item = item.strip().strip('[]"\'')
                if clean_item and len(clean_item) > 2:
                    topics.append(clean_item)
            
            if len(topics) >= 3:
                print(f"=== DEBUG: Fallback parsing worked, got {len(topics)} topics ===")
                return topics[:10]
        
        print("=== DEBUG: AI extraction failed, using fallback topics ===")
        
    except Exception as e:
        print(f"=== DEBUG: AI query failed with error: {str(e)} ===")
        print("=== DEBUG: Using fallback topics ===")
    
    # Return fallback topics
    fallback = fallback_topics.get(document_type, fallback_topics["resume"])
    print(f"=== DEBUG: Returning {len(fallback)} fallback topics ===")
    return fallback


def generate_single_mcq_question(topic, backend, difficulty="easy", document_type="resume"):
    """Generate a single MCQ question on-demand based on document type"""
    
    difficulty_levels = {
        "easy": "basic/fundamental",
        "moderate": "intermediate/practical", 
        "hard": "advanced/expert"
    }
    
    document_context = {
        "resume": f"Create a {difficulty_levels[difficulty]} multiple choice question about '{topic}' suitable for a technical interview. Focus on practical knowledge and real-world applications.",
        
        "research_paper": f"Create a {difficulty_levels[difficulty]} multiple choice question about '{topic}' based on research concepts. Focus on methodologies, findings, and theoretical understanding.",
        
        "technical_document": f"Create a {difficulty_levels[difficulty]} multiple choice question about '{topic}' focusing on technical implementation. Test practical knowledge and technical expertise."
    }
    
    context_prompt = document_context.get(document_type, document_context["resume"])
    
    prompt = f"""
    {context_prompt}
    
    Return ONLY a JSON object in this exact format:
    {{
        "question": "Your question here",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 0,
        "explanation": "Brief explanation of why this is correct"
    }}
    
    The correct_answer should be the index (0-3) of the correct option.
    Make sure the incorrect options are plausible but clearly wrong.
    Ensure the question is relevant to the document type and difficulty level.
    """
    
    try:
        response = llm_query(prompt, backend)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            question_data = json.loads(json_match.group(0))
            return question_data
    except Exception as e:
        print(f"Error generating question: {e}")
        return None
    
    return None


def create_fallback_question(topic, difficulty, document_type):
    """Create a simple fallback question if AI generation fails"""
    
    question_templates = {
        "resume": {
            "easy": f"What is the primary purpose of {topic} in software development?",
            "moderate": f"Which is a best practice when working with {topic}?",
            "hard": f"What is the most critical consideration when optimizing {topic}?"
        },
        "research_paper": {
            "easy": f"What is {topic} primarily used for in research?",
            "moderate": f"Which methodology is commonly associated with {topic}?",
            "hard": f"What are the key limitations of {topic} in current research?"
        },
        "technical_document": {
            "easy": f"What is the main function of {topic}?",
            "moderate": f"How is {topic} typically implemented?",
            "hard": f"What are the scalability considerations for {topic}?"
        }
    }
    
    template = question_templates.get(document_type, question_templates["resume"])
    question = template.get(difficulty, template["easy"])
    
    return {
        "question": question,
        "options": [
            f"Correct answer related to {topic}",
            f"Incorrect option 1",
            f"Incorrect option 2", 
            f"Incorrect option 3"
        ],
        "correct_answer": 0,
        "explanation": f"This tests {difficulty}-level understanding of {topic} in the context of {document_type}."
    }


def tts_speak_to_file(text):
    try:
        engine = pyttsx3.init()
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        engine.save_to_file(text, tmpfile.name)
        engine.runAndWait()
        return tmpfile.name
    except Exception as e:
        raise Exception(f"TTS Error: {str(e)}")


# ==== Routes ====

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start_quiz")
async def start_quiz(
    request: Request, 
    backend: str = Form("gemini"), 
    api_key: str = Form(...),
    document_file: UploadFile = File(...),
    difficulty: str = Form("easy"),
    document_type: str = Form("resume"),
    quiz_type: str = Form("mcq")
):
    try:
        # Validate document type
        valid_types = ["resume", "research_paper", "technical_document"]
        if document_type not in valid_types:
            return JSONResponse({"error": f"Invalid document type. Must be one of: {valid_types}"}, status_code=400)
        
        # Validate API key
        if not api_key or len(api_key.strip()) < 10:
            return JSONResponse({"error": "Please provide a valid API key."}, status_code=400)
        
        # Store API key for this session based on backend
        if backend == "gemini":
            state.gemini_api_key = api_key.strip()
        elif backend == "groq":
            state.groq_api_key = api_key.strip()
        elif backend == "openai":
            state.openai_api_key = api_key.strip()
        else:
            return JSONResponse({"error": "Invalid backend selected."}, status_code=400)
        
        # Extract document text
        document_text = extract_text_from_file(document_file)
        if not document_text or len(document_text.strip()) < 50:
            return JSONResponse({"error": "Document file is empty or too short. Please upload a valid document."}, status_code=400)
        
        # Extract topics and reset state
        state.reset()
        state.document_type = document_type
        state.document_content = document_text[:5000]
        state.backend = backend
        state.difficulty = difficulty
        
        # Re-store the API key after reset
        if backend == "gemini":
            state.gemini_api_key = api_key.strip()
        elif backend == "groq":
            state.groq_api_key = api_key.strip()
        elif backend == "openai":
            state.openai_api_key = api_key.strip()
        
        # Try to extract topics with guaranteed fallback
        try:
            state.topics = extract_document_topics(document_text, document_type, backend)
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            # Use guaranteed fallback topics
            fallback_topics = {
                "resume": ["Technical Skills", "Work Experience", "Problem Solving", "Communication", "Programming", "Database Management", "Web Development", "Project Management"],
                "research_paper": ["Research Methods", "Data Analysis", "Literature Review", "Hypothesis Testing", "Statistical Analysis", "Methodology", "Results Interpretation", "Academic Writing"],
                "technical_document": ["System Architecture", "Implementation", "Best Practices", "Configuration", "Troubleshooting", "API Design", "Documentation", "Technical Specifications"]
            }
            state.topics = fallback_topics.get(document_type, fallback_topics["resume"])
        
        if not state.topics:
            return JSONResponse({"error": "Could not extract topics from document. Please try a different file."}, status_code=400)
        
        return JSONResponse({
            "topics": state.topics, 
            "quiz_type": "mcq",
            "document_type": document_type,
            "message": f"Quiz setup complete! Questions will be generated based on your {document_type.replace('_', ' ').title()}."
        })
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/next_question")
async def next_question(prev_answer: str = Form("")):
    try:
        # Check if quiz is complete
        if state.current_question_index >= state.total_questions:
            return JSONResponse({"question": None, "topic": None})
        
        # Select a random topic from available topics
        if not state.topics:
            return JSONResponse({"question": None, "topic": None})
        
        topic = random.choice(state.topics)
        state.current_topic = topic
        
        # Generate question on-demand
        question_data = generate_single_mcq_question(topic, state.backend, state.difficulty, state.document_type)
        
        # If AI generation fails, use fallback
        if not question_data:
            question_data = create_fallback_question(topic, state.difficulty, state.document_type)
        
        # Store current question data for scoring
        question_data['topic'] = topic
        question_data['question_id'] = state.current_question_index
        state.current_question_data = question_data
        
        return JSONResponse({
            "question": question_data["question"],
            "options": question_data["options"],
            "topic": topic,
            "question_type": "mcq",
            "question_number": state.current_question_index + 1,
            "total_questions": state.total_questions
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/submit_mcq")
async def submit_mcq(selected_answer: int = Form(...)):
    try:
        if not state.current_question_data:
            return JSONResponse({"error": "No current question data available"}, status_code=400)
        
        correct_answer = state.current_question_data["correct_answer"]
        is_correct = selected_answer == correct_answer
        score = 1 if is_correct else 0
        
        # Store the Q&A in history
        state.qa_history.append({
            "topic": state.current_question_data["topic"],
            "question": state.current_question_data["question"],
            "options": state.current_question_data["options"],
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "score": score,
            "explanation": state.current_question_data["explanation"]
        })
        
        state.total_score += score
        state.questions_answered += 1
        state.current_question_index += 1
        
        return JSONResponse({
            "is_correct": is_correct,
            "correct_answer": correct_answer,
            "explanation": state.current_question_data["explanation"],
            "score": score,
            "total_score": state.total_score,
            "max_possible_score": state.total_questions
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/end_quiz")
async def end_quiz():
    """End the quiz early and mark it as completed"""
    try:
        state.quiz_ended_early = True
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/tts")
async def tts(text: str = Form("")):
    try:
        filename = tts_speak_to_file(text)
        return FileResponse(filename, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/final_score")
async def final():
    try:
        # Calculate percentage score
        percentage_score = 0
        if state.questions_answered > 0:
            percentage_score = round((state.total_score / state.questions_answered) * 100, 2)
        
        return JSONResponse({
            "final_score": state.total_score,
            "total_possible": state.total_questions,
            "questions_answered": state.questions_answered,
            "percentage_score": percentage_score,
            "completed_early": state.quiz_ended_early,
            "document_type": state.document_type,
            "history": state.qa_history
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download_results")
async def download_results():
    try:
        if not state.qa_history:
            return JSONResponse({"error": "No quiz data available for download"}, status_code=400)
        
        from fpdf import FPDF
        from datetime import datetime
        
        # Calculate final statistics
        percentage_score = 0
        if state.questions_answered > 0:
            percentage_score = round((state.total_score / state.questions_answered) * 100, 2)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set title
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, 'QUIZ RESULTS REPORT', 0, 1, 'C')
        pdf.ln(10)
        
        # Quiz Summary Section
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'QUIZ SUMMARY', 0, 1, 'L')
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        summary_data = [
            f"Document Type: {state.document_type.replace('_', ' ').title()}",
            f"Difficulty Level: {state.difficulty.title()}",
            f"AI Backend Used: {state.backend.title()}",
            f"Total Questions Available: {state.total_questions}",
            f"Questions Answered: {state.questions_answered}",
            f"Correct Answers: {state.total_score}",
            f"Incorrect Answers: {state.questions_answered - state.total_score}",
            f"Final Score: {percentage_score}%",
            f"Quiz Status: {'Completed Early' if state.quiz_ended_early else 'Completed'}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for item in summary_data:
            pdf.cell(0, 8, item, 0, 1, 'L')
        
        pdf.ln(10)
        
        # Detailed Questions & Answers Section
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'DETAILED QUESTIONS & ANSWERS', 0, 1, 'L')
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        for i, qa in enumerate(state.qa_history, 1):
            # Check if we need a new page
            if pdf.get_y() > 250:
                pdf.add_page()
            
            # Question Header
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f"Question {i}: {qa['topic']}", 0, 1, 'L')
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
            
            # Question Text
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, "Q:", 0, 0, 'L')
            pdf.set_font('Arial', '', 12)
            
            # Handle long questions with text wrapping
            question_text = qa["question"]
            if len(question_text) > 80:
                # Split long questions into multiple lines
                words = question_text.split(' ')
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                lines.append(current_line.strip())
                
                for j, line in enumerate(lines):
                    if j == 0:
                        pdf.cell(0, 8, f" {line}", 0, 1, 'L')
                    else:
                        pdf.cell(0, 8, f"   {line}", 0, 1, 'L')
            else:
                pdf.cell(0, 8, f" {question_text}", 0, 1, 'L')
            
            pdf.ln(3)
            
            # Options
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, "Options:", 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            
            for j, option in enumerate(qa["options"]):
                option_letter = chr(65 + j)
                # Truncate very long options
                if len(option) > 70:
                    option = option[:67] + "..."
                pdf.cell(0, 6, f"   {option_letter}) {option}", 0, 1, 'L')
            
            pdf.ln(2)
            
            # Answers and Result
            pdf.set_font('Arial', 'B', 12)
            your_answer = chr(65 + qa["selected_answer"]) if qa["selected_answer"] is not None else "No answer"
            correct_answer = chr(65 + qa["correct_answer"])
            result = "Correct ✓" if qa["is_correct"] else "Incorrect ✗"
            
            pdf.cell(0, 6, f"Your Answer: {your_answer}", 0, 1, 'L')
            pdf.cell(0, 6, f"Correct Answer: {correct_answer}", 0, 1, 'L')
            
            # Color-code the result
            if qa["is_correct"]:
                pdf.set_text_color(0, 128, 0)  # Green for correct
            else:
                pdf.set_text_color(255, 0, 0)  # Red for incorrect
            
            pdf.cell(0, 6, f"Result: {result}", 0, 1, 'L')
            pdf.set_text_color(0, 0, 0)  # Reset to black
            
            pdf.cell(0, 6, f"Points Earned: {qa['score']}/1", 0, 1, 'L')
            pdf.ln(2)
            
            # Explanation
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 6, "Explanation:", 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            
            # Handle long explanations
            explanation = qa["explanation"]
            if len(explanation) > 80:
                words = explanation.split(' ')
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                lines.append(current_line.strip())
                
                for line in lines:
                    pdf.cell(0, 6, f"   {line}", 0, 1, 'L')
            else:
                pdf.cell(0, 6, f"   {explanation}", 0, 1, 'L')
            
            pdf.ln(8)
        
        # Final Summary Page
        if pdf.get_y() > 200:
            pdf.add_page()
        
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'END OF REPORT', 0, 1, 'C')
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Total Score: {state.total_score}/{state.questions_answered} ({percentage_score}%)", 0, 1, 'C')
        
        # Create temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        # Save PDF to temporary file
        pdf.output(temp_file.name)
        
        # Return file for download
        filename = f"quiz_results_{state.document_type}_{state.difficulty}.pdf"
        
        return FileResponse(
            temp_file.name,
            media_type="application/pdf",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ==== Run Server ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
