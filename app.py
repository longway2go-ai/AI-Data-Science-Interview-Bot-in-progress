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
import openai  # New import for OpenAI
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

# ==== Interview State ====
class InterviewState:
    def __init__(self):
        self.topics = []
        self.current_question_index = 0
        self.qa_history = []
        self.total_score = 0
        self.backend = "gemini"
        self.gemini_api_key = ""
        self.groq_api_key = ""
        self.openai_api_key = ""  # New for OpenAI
        self.total_questions = 30
        self.questions_answered = 0
        self.interview_ended_early = False
        self.current_topic = ""
        self.difficulty = "easy"
        self.current_question_data = None

    def reset(self):
        self.topics = []
        self.current_question_index = 0
        self.qa_history = []
        self.total_score = 0
        self.total_questions = 30
        self.questions_answered = 0
        self.interview_ended_early = False
        self.current_topic = ""
        self.difficulty = "easy"
        self.current_question_data = None

state = InterviewState()

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
            model="gpt-4o-mini",  # Using the more affordable model
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
def extract_resume_topics(resume_text, backend):
    prompt = f"""
    Analyze this resume and extract 8-10 main technical topics, skills, or job roles that would be good for interview questions.
    Focus on technical skills, programming languages, frameworks, tools, and job responsibilities.
    Return only a Python list format like: ['Python', 'Machine Learning', 'API Development', 'Database Design']
    
    Resume content:
    {resume_text[:2000]}...
    """
    
    try:
        topics_text = llm_query(prompt, backend)
        # Try to extract a list from the response
        list_match = re.search(r'\[.*?\]', topics_text, re.DOTALL)
        if list_match:
            try:
                parsed = eval(list_match.group(0))
                return parsed[:10]  # Limit to 10 topics max
            except:
                pass
        
        # Fallback: split by commas and clean up
        topics = [t.strip().strip('"\'') for t in topics_text.split(',') if t.strip()]
        return topics[:10]
        
    except Exception as e:
        raise Exception(f"Error extracting topics: {str(e)}")

def generate_single_mcq_question(topic, backend, difficulty="easy"):
    """Generate a single MCQ question on-demand"""
    difficulty_prompts = {
        "easy": f"Create a basic multiple choice question about '{topic}' suitable for beginners. Focus on fundamental concepts.",
        "moderate": f"Create an intermediate multiple choice question about '{topic}' that requires practical knowledge.",
        "hard": f"Create an advanced multiple choice question about '{topic}' that tests deep expertise."
    }
    
    prompt = f"""
    {difficulty_prompts.get(difficulty, difficulty_prompts["easy"])}
    
    Return ONLY a JSON object in this exact format:
    {{
        "question": "Your question here",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 0,
        "explanation": "Brief explanation of why this is correct"
    }}
    
    The correct_answer should be the index (0-3) of the correct option.
    Make sure the incorrect options are plausible but clearly wrong.
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

def create_fallback_question(topic, difficulty):
    """Create a simple fallback question if AI generation fails"""
    difficulty_questions = {
        "easy": {
            "question": f"What is the primary purpose of {topic}?",
            "options": [
                f"To provide essential functionality",
                f"To complicate the system",
                f"To reduce performance", 
                f"To increase errors"
            ]
        },
        "moderate": {
            "question": f"Which is a best practice when working with {topic}?",
            "options": [
                f"Following standard conventions",
                f"Ignoring documentation",
                f"Using deprecated methods",
                f"Avoiding error handling"
            ]
        },
        "hard": {
            "question": f"What is the most critical consideration when optimizing {topic}?",
            "options": [
                f"Performance and scalability",
                f"Code aesthetics only",
                f"Using the simplest approach",
                f"Avoiding all optimizations"
            ]
        }
    }
    
    fallback = difficulty_questions.get(difficulty, difficulty_questions["easy"])
    return {
        "question": fallback["question"],
        "options": fallback["options"],
        "correct_answer": 0,
        "explanation": f"This tests {difficulty}-level understanding of {topic}."
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

@app.post("/start_interview")
async def start_interview(
    request: Request, 
    backend: str = Form("gemini"), 
    api_key: str = Form(...),
    resume_file: UploadFile = File(...),
    difficulty: str = Form("easy"),
    interview_type: str = Form("mcq")
):
    try:
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
        
        # Extract resume text
        resume_text = extract_text_from_file(resume_file)
        if not resume_text or len(resume_text.strip()) < 50:
            return JSONResponse({"error": "Resume file is empty or too short. Please upload a valid resume."}, status_code=400)
        
        # Extract topics and reset state
        state.reset()
        state.topics = extract_resume_topics(resume_text, backend)
        state.backend = backend
        state.difficulty = difficulty
        
        # Re-store the API key after reset
        if backend == "gemini":
            state.gemini_api_key = api_key.strip()
        elif backend == "groq":
            state.groq_api_key = api_key.strip()
        elif backend == "openai":
            state.openai_api_key = api_key.strip()
        
        if not state.topics:
            return JSONResponse({"error": "Could not extract topics from resume. Please try a different file."}, status_code=400)
        
        # Set total questions but don't pre-generate them
        state.total_questions = 40
        
        return JSONResponse({
            "topics": state.topics, 
            "interview_type": "mcq",
            "message": "Interview setup complete! Questions will be generated as you progress."
        })
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/next_question")
async def next_question(prev_answer: str = Form("")):
    try:
        # Check if interview is complete
        if state.current_question_index >= state.total_questions:
            return JSONResponse({"question": None, "topic": None})
        
        # Select a random topic from available topics
        if not state.topics:
            return JSONResponse({"question": None, "topic": None})
        
        topic = random.choice(state.topics)
        state.current_topic = topic
        
        # Generate question on-demand
        question_data = generate_single_mcq_question(topic, state.backend, state.difficulty)
        
        # If AI generation fails, use fallback
        if not question_data:
            question_data = create_fallback_question(topic, state.difficulty)
        
        # Store current question data for scoring
        question_data['topic'] = topic
        question_data['question_id'] = state.current_question_index
        state.current_question_data = question_data
        
        return JSONResponse({
            "question": question_data["question"],
            "options": question_data["options"],
            "topic": topic,
            "question_type": "mcq"
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
        score = 10 if is_correct else 0
        
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
            "total_score": state.total_score
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/end_interview")
async def end_interview():
    """End the interview early and mark it as completed"""
    try:
        state.interview_ended_early = True
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
        # Calculate average score if there are answered questions
        average_score = 0
        if state.questions_answered > 0:
            average_score = round(state.total_score / state.questions_answered, 2)
        
        return JSONResponse({
            "final_score": state.total_score,
            "average_score": average_score,
            "total_questions": state.total_questions,
            "questions_answered": state.questions_answered,
            "completed_early": state.interview_ended_early,
            "history": state.qa_history
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==== Run Server ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
