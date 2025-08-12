import gradio as gr
import json
import time
import threading
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize AI models
try:
    # Use lightweight models for better performance
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # For text analysis and scoring
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # For question answering evaluation
    qa_evaluator = pipeline("question-answering", 
                           model="distilbert-base-cased-distilled-squad")
except Exception as e:
    print(f"Model loading error: {e}")
    # Fallback to None if models fail to load
    model_name, tokenizer, model, sentiment_analyzer, qa_evaluator = None, None, None, None, None

class TimedInterviewBot:
    def __init__(self):
        self.reset_interview()
        
        # Interview configuration
        self.INTERVIEW_DURATION = 15 * 60  # 15 minutes in seconds
        self.MIN_QUESTIONS = 5  # Minimum questions to ask
        self.MAX_QUESTIONS = 12  # Maximum questions available
        
        self.question_templates = {
            'opening': [
                "Tell me about yourself and your journey into data science.",
                "What sparked your interest in data science and machine learning?",
                "How would you explain data science to a non-technical stakeholder?"
            ],
            'projects': [
                "Walk me through your most impactful data science project from start to finish.",
                "Describe a project where you had to deal with messy or incomplete data. How did you handle it?",
                "Tell me about a machine learning model you built. What was the business impact?",
                "Explain a time when your initial model didn't perform well. How did you improve it?",
                "Describe a project where you had to work with large datasets. What challenges did you face?"
            ],
            'technical_skills': [
                "Which programming languages do you prefer for data science and why?",
                "How do you approach feature engineering and selection in your projects?",
                "Explain your experience with different machine learning algorithms and when to use them.",
                "What data visualization tools do you use and how do you choose the right chart type?",
                "How do you handle missing data in your datasets?"
            ],
            'ml_concepts': [
                "Explain the bias-variance tradeoff and how it affects model performance.",
                "What's the difference between supervised, unsupervised, and reinforcement learning?",
                "How do you prevent overfitting in machine learning models?",
                "Explain cross-validation and why it's important in model evaluation.",
                "What metrics would you use to evaluate a classification vs regression model?",
                "Describe the difference between bagging and boosting algorithms."
            ],
            'data_processing': [
                "How do you approach exploratory data analysis (EDA) for a new dataset?",
                "Explain your process for data cleaning and preprocessing.",
                "How do you handle outliers in your data?",
                "What's your approach to feature scaling and normalization?",
                "How do you deal with imbalanced datasets?"
            ],
            'business_impact': [
                "How do you ensure your data science work aligns with business objectives?",
                "Describe a time when you had to present technical findings to non-technical stakeholders.",
                "How do you measure the success of a data science project?",
                "Tell me about a project where your analysis changed a business decision.",
                "How do you handle situations where data doesn't support the expected hypothesis?"
            ],
            'tools_platforms': [
                "What's your experience with cloud platforms like AWS, Azure, or GCP for data science?",
                "How do you approach model deployment and monitoring in production?",
                "Explain your experience with SQL and database management for data projects.",
                "What version control and collaboration tools do you use for data science projects?",
                "How do you ensure reproducibility in your data science workflows?"
            ],
            'achievements': [
                "What's your proudest achievement in data science so far?",
                "Describe a time when you solved a particularly challenging analytical problem.",
                "Tell me about recognition or awards you've received for your data science work.",
                "What's the most innovative approach you've taken to solve a data problem?"
            ]
        }
        
    def reset_interview(self):
        """Reset all interview data for new session"""
        self.interview_data = {
            'questions': [],
            'answers': [],
            'timestamps': [],
            'scores': [],
            'total_score': 0,
            'feedback': [],
            'resume_content': '',
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'is_active': False,
            'force_ended': False
        }
        
        self.current_question_index = 0
        self.timer_thread = None
        self.remaining_time = self.INTERVIEW_DURATION
        self.time_callbacks = []
        
    def start_timer(self, update_callback=None):
        """Start the 15-minute countdown timer"""
        self.interview_data['start_time'] = datetime.now()
        self.interview_data['is_active'] = True
        self.remaining_time = self.INTERVIEW_DURATION
        
        if update_callback:
            self.time_callbacks.append(update_callback)
        
        def timer_worker():
            while self.remaining_time > 0 and self.interview_data['is_active']:
                time.sleep(1)
                self.remaining_time -= 1
                
                # Notify callbacks of time update
                for callback in self.time_callbacks:
                    try:
                        callback(self.remaining_time)
                    except:
                        pass
            
            # Time's up - force end interview
            if self.remaining_time <= 0:
                self.force_end_interview("Time limit reached")
        
        self.timer_thread = threading.Thread(target=timer_worker, daemon=True)
        self.timer_thread.start()
    
    def force_end_interview(self, reason=""):
        """Force end the interview due to time limit or user exit"""
        self.interview_data['is_active'] = False
        self.interview_data['force_ended'] = True
        self.interview_data['end_time'] = datetime.now()
        
        if self.interview_data['start_time']:
            duration = self.interview_data['end_time'] - self.interview_data['start_time']
            self.interview_data['duration_seconds'] = duration.total_seconds()
    
    def get_time_remaining_formatted(self) -> str:
        """Get formatted time remaining string"""
        minutes = self.remaining_time // 60
        seconds = self.remaining_time % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def analyze_resume(self, resume_text: str) -> Dict:
        """Analyze resume content and generate personalized questions"""
        if not resume_text:
            return {"error": "No resume content provided"}
            
        self.interview_data['resume_content'] = resume_text
        
        # Extract key information from resume
        skills = self.extract_skills(resume_text)
        experience_years = self.extract_experience(resume_text)
        roles = self.extract_roles(resume_text)
        
        # Generate personalized questions based on resume
        personalized_questions = self.generate_personalized_questions(skills, experience_years, roles)
        
        return {
            "skills": skills,
            "experience_years": experience_years,
            "roles": roles,
            "personalized_questions": personalized_questions
        }
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract data science skills from resume"""
        ds_skills = [
            # Programming Languages
            'python', 'r', 'sql', 'scala', 'java', 'julia', 'matlab',
            
            # Machine Learning & AI
            'machine learning', 'deep learning', 'neural networks', 'cnn', 'rnn', 'lstm',
            'natural language processing', 'nlp', 'computer vision', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'classification', 'regression',
            'clustering', 'dimensionality reduction', 'feature engineering',
            
            # ML Libraries & Frameworks  
            'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'pandas', 'numpy',
            'scipy', 'matplotlib', 'seaborn', 'plotly', 'opencv', 'nltk', 'spacy',
            'xgboost', 'lightgbm', 'catboost', 'hugging face', 'transformers',
            
            # Big Data & Cloud
            'spark', 'hadoop', 'kafka', 'airflow', 'aws', 'azure', 'gcp', 'sagemaker',
            'databricks', 'snowflake', 'bigquery', 'redshift', 'docker', 'kubernetes',
            
            # Databases
            'postgresql', 'mysql', 'mongodb', 'cassandra', 'elasticsearch', 'redis',
            
            # Statistics & Math
            'statistics', 'probability', 'linear algebra', 'calculus', 'hypothesis testing',
            'a/b testing', 'experimental design', 'bayesian statistics',
            
            # Visualization & BI
            'tableau', 'power bi', 'looker', 'qlik', 'd3.js', 'jupyter', 'colab',
            
            # MLOps & Deployment
            'mlflow', 'kubeflow', 'mlops', 'model deployment', 'api development', 'flask', 'fastapi',
            
            # Domain Specific
            'time series', 'forecasting', 'recommendation systems', 'anomaly detection',
            'sentiment analysis', 'image processing', 'data mining', 'web scraping'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in ds_skills if skill in text_lower]
        return found_skills[:12]  # Limit to top 12 skills
    
    def extract_experience(self, text: str) -> int:
        """Extract years of experience from resume"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*year\s*(?:of\s*)?(?:professional\s*)?experience'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        return 0
    
    def extract_roles(self, text: str) -> List[str]:
        """Extract data science roles from resume"""
        ds_roles = [
            'data scientist', 'machine learning engineer', 'ml engineer', 'data analyst',
            'data engineer', 'research scientist', 'ai engineer', 'business analyst',
            'quantitative analyst', 'statistician', 'data science intern', 'ml intern',
            'research assistant', 'data science researcher', 'ai researcher',
            'senior data scientist', 'principal data scientist', 'lead data scientist',
            'data science manager', 'head of data science', 'chief data officer',
            'nlp engineer', 'computer vision engineer', 'deep learning engineer',
            'analytics consultant', 'data consultant', 'freelancer', 'contractor'
        ]
        
        text_lower = text.lower()
        found_roles = [role for role in ds_roles if role in text_lower]
        return found_roles[:6]  # Limit to top 6 roles
    
    def generate_personalized_questions(self, skills: List[str], experience: int, roles: List[str]) -> List[str]:
        """Generate personalized questions based on data science resume analysis"""
        questions = []
        
        # Experience-based questions for data science
        if experience > 5:
            questions.extend([
                "With your extensive data science experience, how do you approach building scalable ML pipelines?",
                "How do you mentor junior data scientists and guide them through complex projects?",
                "What's the most challenging data science problem you've solved in your career?"
            ])
        elif experience > 2:
            questions.extend([
                "How have you evolved your data science methodology over the past few years?",
                "Describe a time when you had to learn a new ML technique or tool quickly for a project.",
                "What's been your biggest learning experience in data science recently?"
            ])
        else:
            questions.extend([
                "What initially drew you to data science and what keeps you passionate about it?",
                "How do you stay current with the rapidly evolving data science landscape?",
                "Tell me about a data science project from your studies or bootcamp that you're proud of."
            ])
        
        # Skill-specific questions
        ml_frameworks = [s for s in skills if s in ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn']]
        if ml_frameworks:
            questions.append(f"I see you have experience with {ml_frameworks[0]}. Can you walk me through how you've used it in a recent project?")
        
        if 'deep learning' in skills or 'neural networks' in skills:
            questions.append("Explain a deep learning project you've worked on. What architecture did you choose and why?")
        
        if 'nlp' in skills or 'natural language processing' in skills:
            questions.append("Describe your experience with NLP. What techniques have you used for text preprocessing and modeling?")
        
        if 'computer vision' in skills or 'opencv' in skills:
            questions.append("Tell me about a computer vision project you've implemented. What were the main challenges?")
        
        # Cloud and deployment questions
        cloud_skills = [s for s in skills if s in ['aws', 'azure', 'gcp', 'sagemaker', 'databricks']]
        if cloud_skills:
            questions.append(f"How have you used {cloud_skills[0]} for data science projects? Any experience with model deployment?")
        
        # Big data questions
        big_data_skills = [s for s in skills if s in ['spark', 'hadoop', 'kafka', 'bigquery', 'snowflake']]
        if big_data_skills:
            questions.append(f"Describe your experience working with {big_data_skills[0]} for large-scale data processing.")
        
        # Role-specific questions
        if 'machine learning engineer' in roles or 'ml engineer' in roles:
            questions.append("What's your approach to productionizing machine learning models? How do you handle model monitoring?")
        
        if 'data engineer' in roles:
            questions.append("How do you design data pipelines that serve both analytics and ML use cases?")
        
        if 'research scientist' in roles or 'ai researcher' in roles:
            questions.append("Tell me about a research project you've worked on. How do you balance theoretical rigor with practical applications?")
        
        return questions[:5]  # Limit to 5 personalized questions
    
    def get_next_question(self) -> Tuple[str, bool]:
        """Get the next interview question and whether interview should continue"""
        if not self.interview_data['is_active']:
            return "Interview has ended.", False
        
        # Build question queue based on time remaining and questions answered
        all_questions = []
        
        # Add personalized questions first if available
        if hasattr(self, 'personalized_questions'):
            all_questions.extend(self.personalized_questions)
        
        # Add template questions based on interview stage and DS focus
        questions_answered = len(self.interview_data['questions'])
        time_progress = 1 - (self.remaining_time / self.INTERVIEW_DURATION)
        
        if questions_answered < 2:
            # Opening questions
            all_questions.extend(self.question_templates['opening'])
        elif time_progress < 0.3:
            # Early stage - focus on projects and technical skills
            all_questions.extend(self.question_templates['projects'])
            all_questions.extend(self.question_templates['technical_skills'])
        elif time_progress < 0.6:
            # Middle stage - ML concepts and data processing
            all_questions.extend(self.question_templates['ml_concepts'])
            all_questions.extend(self.question_templates['data_processing'])
        elif time_progress < 0.8:
            # Later stage - business impact and tools
            all_questions.extend(self.question_templates['business_impact'])
            all_questions.extend(self.question_templates['tools_platforms'])
        else:
            # Final stage - achievements and closing
            all_questions.extend(self.question_templates['achievements'])
            all_questions.extend([
                "Where do you see the future of data science heading in the next few years?",
                "What questions do you have about our data science team and projects?"
            ])
        
        # Check if we should continue
        should_continue = (
            self.current_question_index < len(all_questions) and
            self.remaining_time > 30 and  # At least 30 seconds remaining
            questions_answered < self.MAX_QUESTIONS and
            self.interview_data['is_active']
        )
        
        if should_continue and self.current_question_index < len(all_questions):
            question = all_questions[self.current_question_index]
            self.current_question_index += 1
            self.interview_data['questions'].append(question)
            return question, True
        else:
            # End interview
            self.force_end_interview("Natural conclusion")
            return "Thank you for completing the interview! Let me generate your detailed performance report.", False
    
    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """Evaluate user's answer with comprehensive scoring"""
        if not answer.strip():
            return {
                "score": 0,
                "feedback": "No answer provided.",
                "strengths": [],
                "improvements": ["Please provide a complete answer to demonstrate your communication skills."],
                "detailed_scores": {}
            }
        
        # Multi-dimensional scoring
        score_components = {
            "content_quality": self.score_content_quality(answer),
            "communication": self.score_communication_skills(answer),
            "relevance": self.score_relevance(question, answer),
            "professionalism": self.score_professionalism(answer),
            "specificity": self.score_specificity(answer),
            "structure": self.score_structure(answer)
        }
        
        # Weighted scoring system
        weights = {
            "content_quality": 0.25,
            "communication": 0.20,
            "relevance": 0.20,
            "professionalism": 0.15,
            "specificity": 0.15,
            "structure": 0.05
        }
        
        # Calculate weighted total score
        total_score = sum(score_components[component] * weights[component] 
                         for component in score_components)
        total_score = min(100, max(0, total_score))
        
        # Generate comprehensive feedback
        feedback = self.generate_comprehensive_feedback(total_score, score_components, answer, question)
        
        # Store evaluation data
        self.interview_data['answers'].append(answer)
        self.interview_data['scores'].append(total_score)
        self.interview_data['feedback'].append(feedback)
        self.interview_data['timestamps'].append(datetime.now().isoformat())
        
        return {
            "score": round(total_score, 1),
            "feedback": feedback['overall'],
            "strengths": feedback['strengths'],
            "improvements": feedback['improvements'],
            "detailed_scores": score_components
        }
    
    def score_content_quality(self, answer: str) -> float:
        """Score the quality and depth of content"""
        word_count = len(answer.split())
        
        # Optimal length scoring
        if 40 <= word_count <= 120:
            length_score = 100
        elif 20 <= word_count < 40:
            length_score = 70
        elif word_count < 20:
            length_score = 40
        else:
            length_score = 85  # Long but acceptable
        
        # Content depth indicators
        depth_indicators = [
            'because', 'therefore', 'however', 'although', 'specifically',
            'for instance', 'such as', 'resulted in', 'led to', 'achieved'
        ]
        
        depth_score = min(100, 60 + sum(10 for indicator in depth_indicators 
                                      if indicator in answer.lower()))
        
        return (length_score * 0.4) + (depth_score * 0.6)
    
    def score_communication_skills(self, answer: str) -> float:
        """Score communication clarity and flow"""
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        # Sentence structure scoring
        if 2 <= len(sentences) <= 6:
            structure_score = 90
        elif len(sentences) == 1:
            structure_score = 60
        else:
            structure_score = 75
        
        # Transition words indicating good flow
        transitions = ['first', 'then', 'next', 'finally', 'additionally', 'furthermore', 'moreover']
        transition_score = min(100, 70 + sum(5 for t in transitions if t in answer.lower()))
        
        return (structure_score * 0.6) + (transition_score * 0.4)
    
    def score_relevance(self, question: str, answer: str) -> float:
        """Score how well the answer addresses the question"""
        question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
        answer_keywords = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        question_keywords -= stop_words
        answer_keywords -= stop_words
        
        if not question_keywords:
            return 75  # Default score when no keywords found
        
        overlap = len(question_keywords.intersection(answer_keywords))
        relevance_ratio = overlap / len(question_keywords)
        
        # Base score plus relevance bonus
        return min(100, 50 + (relevance_ratio * 50))
    
    def score_professionalism(self, answer: str) -> float:
        """Score professional tone and language"""
        try:
            if sentiment_analyzer:
                sentiment = sentiment_analyzer(answer)[0]
                
                # Professional indicators
                professional_phrases = [
                    'in my experience', 'i believe', 'i would', 'my approach',
                    'i learned', 'i developed', 'i managed', 'i collaborated'
                ]
                
                professional_count = sum(1 for phrase in professional_phrases 
                                       if phrase in answer.lower())
                
                # Base score from sentiment
                if sentiment['label'] == 'LABEL_2':  # Positive
                    base_score = 80 + (sentiment['score'] * 20)
                elif sentiment['label'] == 'LABEL_1':  # Neutral
                    base_score = 70 + (sentiment['score'] * 15)
                else:  # Negative but might be discussing challenges professionally
                    base_score = 60
                
                # Boost for professional language
                professional_boost = min(20, professional_count * 5)
                
                return min(100, base_score + professional_boost)
            else:
                return 75
        except:
            return 75
    
    def score_specificity(self, answer: str) -> float:
        """Score use of specific data science examples and technical details"""
        # Data science specific indicators
        ds_specificity_indicators = [
            # Project terms
            'project', 'dataset', 'model', 'algorithm', 'pipeline', 'feature', 
            'training', 'validation', 'testing', 'deployment', 'production',
            
            # Technical terms
            'accuracy', 'precision', 'recall', 'f1-score', 'auc', 'rmse', 'mae',
            'cross-validation', 'overfitting', 'underfitting', 'hyperparameter',
            'preprocessing', 'feature engineering', 'dimensionality reduction',
            
            # Tools and methods
            'python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
            'pytorch', 'jupyter', 'git', 'aws', 'docker', 'api',
            
            # Business impact
            'improved', 'increased', 'reduced', 'optimized', 'automated',
            'achieved', 'delivered', 'implemented', 'developed', 'built',
            'analyzed', 'predicted', 'classified', 'clustered', 'detected'
        ]
        
        # Numbers, percentages, and metrics
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', answer)
        
        # Technical metrics patterns
        metric_patterns = [
            r'\b\d+(?:\.\d+)?%\s*(?:accuracy|precision|recall|improvement)',
            r'\b\d+(?:\.\d+)?\s*(?:gb|tb|million|thousand)\s*(?:records|rows|samples)',
            r'\b\d+(?:\.\d+)?\s*(?:days|weeks|months)\s*(?:faster|reduction|improvement)'
        ]
        
        metric_matches = sum(len(re.findall(pattern, answer.lower())) for pattern in metric_patterns)
        
        specificity_count = sum(1 for indicator in ds_specificity_indicators 
                              if indicator in answer.lower())
        
        # Base score from DS-specific terms
        base_score = min(85, 35 + (specificity_count * 6))
        
        # Bonus for metrics and numbers
        numbers_bonus = min(15, len(numbers) * 4)
        metric_bonus = min(10, metric_matches * 5)
        
        return base_score + numbers_bonus + metric_bonus
    
    def score_structure(self, answer: str) -> float:
        """Score answer organization and logical flow"""
        # Check for clear structure indicators
        structure_words = ['first', 'second', 'third', 'initially', 'then', 'finally', 'in conclusion']
        structure_count = sum(1 for word in structure_words if word in answer.lower())
        
        # Paragraph-like structure (multiple sentences)
        sentences = [s for s in answer.split('.') if s.strip()]
        
        if structure_count >= 2:
            return 95
        elif structure_count >= 1:
            return 85
        elif len(sentences) >= 3:
            return 75
        else:
            return 60
    
    def generate_comprehensive_feedback(self, score: float, score_components: Dict, 
                                     answer: str, question: str) -> Dict:
        """Generate detailed, actionable feedback"""
        strengths = []
        improvements = []
        
        # Analyze each scoring component
        for component, component_score in score_components.items():
            if component_score >= 80:
                if component == "content_quality":
                    strengths.append("Provided comprehensive and well-developed content")
                elif component == "communication":
                    strengths.append("Demonstrated clear and effective communication skills")
                elif component == "relevance":
                    strengths.append("Directly addressed the question with relevant information")
                elif component == "professionalism":
                    strengths.append("Maintained professional tone and language throughout")
                elif component == "specificity":
                    strengths.append("Used concrete examples and specific details effectively")
                elif component == "structure":
                    strengths.append("Organized response with clear logical flow")
            elif component_score < 70:
                if component == "content_quality":
                    improvements.append("Provide more detailed and comprehensive responses")
                elif component == "communication":
                    improvements.append("Focus on clearer sentence structure and flow")
                elif component == "relevance":
                    improvements.append("Ensure your answer directly addresses the question asked")
                elif component == "professionalism":
                    improvements.append("Use more professional language and positive framing")
                elif component == "specificity":
                    improvements.append("Include specific examples, numbers, and concrete details")
                elif component == "structure":
                    improvements.append("Organize your thoughts with clearer structure")
        
        # Overall performance feedback
        if score >= 90:
            overall = "Outstanding response! You demonstrated excellent interview skills with comprehensive, well-structured, and professional communication."
        elif score >= 80:
            overall = "Very strong answer with good content and professional delivery. Minor refinements could make it even better."
        elif score >= 70:
            overall = "Solid response with good foundation. Focus on the improvement areas to enhance your interview performance."
        elif score >= 60:
            overall = "Acceptable answer but needs development. Work on providing more detailed and structured responses."
        else:
            overall = "This answer needs significant improvement. Focus on being more comprehensive, specific, and relevant to the question."
        
        return {
            "overall": overall,
            "strengths": strengths,
            "improvements": improvements
        }
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive final interview report"""
        if not self.interview_data['scores']:
            return {"error": "No interview data available for report generation"}
        
        # Calculate performance metrics
        avg_score = sum(self.interview_data['scores']) / len(self.interview_data['scores'])
        highest_score = max(self.interview_data['scores'])
        lowest_score = min(self.interview_data['scores'])
        
        # Time analysis
        duration_minutes = self.interview_data.get('duration_seconds', 0) / 60
        questions_answered = len(self.interview_data['questions'])
        
        # Performance categorization with detailed criteria
        if avg_score >= 85:
            performance_level = "Outstanding"
            performance_description = "Exceptional interview performance demonstrating strong professional communication and comprehensive responses."
            recommendation = "Highly recommended candidate with excellent interview skills"
        elif avg_score >= 75:
            performance_level = "Strong" 
            performance_description = "Solid interview performance with good communication skills and relevant content delivery."
            recommendation = "Recommended candidate with strong potential"
        elif avg_score >= 65:
            performance_level = "Satisfactory"
            performance_description = "Adequate interview performance with room for improvement in several key areas."
            recommendation = "Candidate shows potential but needs development in interview skills"
        else:
            performance_level = "Needs Improvement"
            performance_description = "Interview performance requires significant development across multiple areas."
            recommendation = "Candidate should focus on interview preparation and practice before future opportunities"
        
        # Collect and analyze feedback patterns
        all_strengths = []
        all_improvements = []
        for feedback in self.interview_data['feedback']:
            all_strengths.extend(feedback.get('strengths', []))
            all_improvements.extend(feedback.get('improvements', []))
        
        # Get most common feedback themes
        unique_strengths = list(dict.fromkeys(all_strengths))[:7]
        unique_improvements = list(dict.fromkeys(all_improvements))[:7]
        
        # Calculate consistency score (low standard deviation = more consistent)
        if len(self.interview_data['scores']) > 1:
            import statistics
            consistency = 100 - min(30, statistics.stdev(self.interview_data['scores']))
        else:
            consistency = 100
        
        report = {
            "interview_summary": {
                "overall_score": round(avg_score, 1),
                "performance_level": performance_level,
                "performance_description": performance_description,
                "recommendation": recommendation,
                "consistency_score": round(consistency, 1)
            },
            "session_details": {
                "questions_answered": questions_answered,
                "duration_minutes": round(duration_minutes, 1),
                "completion_status": "Completed" if not self.interview_data.get('force_ended') else "Time Limited",
                "highest_score": round(highest_score, 1),
                "lowest_score": round(lowest_score, 1),
                "score_range": round(highest_score - lowest_score, 1)
            },
            "key_insights": {
                "strengths": unique_strengths,
                "areas_for_improvement": unique_improvements,
                "score_trend": "Improving" if len(self.interview_data['scores']) > 1 and 
                             self.interview_data['scores'][-1] > self.interview_data['scores'][0] else "Stable"
            },
            "detailed_breakdown": [
                {
                    "question_number": i + 1,
                    "question": q,
                    "score": s,
                    "feedback": f.get('overall', ''),
                    "word_count": len(self.interview_data['answers'][i].split()) if i < len(self.interview_data['answers']) else 0
                }
                for i, (q, s, f) in enumerate(zip(
                    self.interview_data['questions'],
                    self.interview_data['scores'],
                    self.interview_data['feedback']
                ))
            ],
            "recommendations": self.generate_personalized_recommendations(avg_score, unique_improvements),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "interview_version": "1.0",
                "total_possible_score": 100
            }
        }
        
        return report
    
    def generate_personalized_recommendations(self, avg_score: float, improvements: List[str]) -> List[str]:
        """Generate personalized recommendations for data science interviews"""
        recommendations = []
        
        if avg_score < 60:
            recommendations.extend([
                "Practice explaining data science projects using the STAR method (Situation, Task, Action, Result)",
                "Prepare 3-5 detailed project examples covering different aspects: EDA, modeling, deployment, business impact",
                "Study common data science interview questions and practice technical explanations",
                "Review fundamental ML concepts: bias-variance tradeoff, overfitting, cross-validation, evaluation metrics"
            ])
        elif avg_score < 75:
            recommendations.extend([
                "Focus on quantifying your project impacts with specific metrics (accuracy improvements, processing time reductions, cost savings)",
                "Practice explaining complex ML concepts in simple terms for both technical and non-technical audiences",
                "Prepare examples that demonstrate your end-to-end data science workflow from problem definition to deployment",
                "Study advanced topics relevant to your target role: MLOps, model monitoring, A/B testing, experimentation"
            ])
        else:
            recommendations.extend([
                "Practice discussing cutting-edge techniques and how they apply to business problems",
                "Prepare thoughtful questions about the company's data infrastructure, ML stack, and data science challenges",
                "Focus on demonstrating thought leadership: how you stay current, contribute to the DS community, mentor others",
                "Practice case study discussions where you can showcase strategic thinking about data science initiatives"
            ])
        
        # Add specific recommendations based on common improvement areas
        improvement_text = ' '.join(improvements).lower()
        
        if "specific" in improvement_text or "detail" in improvement_text:
            recommendations.append("Prepare a 'project portfolio' with 3-4 detailed case studies including: problem statement, data description, methodology, results, and business impact")
        
        if "structure" in improvement_text or "organize" in improvement_text:
            recommendations.append("Use the CRISP-DM or similar framework to structure your project discussions: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment")
        
        if "professional" in improvement_text or "communication" in improvement_text:
            recommendations.append("Practice explaining technical concepts with business context: 'I used random forests because they provide good interpretability for stakeholder buy-in while maintaining high accuracy'")
        
        if "relevance" in improvement_text:
            recommendations.append("Listen carefully to questions and address all parts: if asked about a challenging project, make sure to cover the challenge, your approach, and the outcome")
        
        # Data science specific recommendations
        recommendations.extend([
            "Practice coding questions: SQL queries, Python/R data manipulation, basic algorithm implementation",
            "Be ready to discuss trade-offs: model complexity vs interpretability, accuracy vs speed, batch vs real-time processing",
            "Prepare to discuss data ethics, bias in ML models, and responsible AI practices"
        ])
        
        return recommendations[:6]  # Limit to top 6 recommendations

# Global interview bot instance
interview_bot = TimedInterviewBot()

# Gradio interface functions
def start_interview(resume_file):
    """Initialize interview with resume upload and start timer"""
    global interview_bot
    interview_bot.reset_interview()  # Reset for new interview
    
    if resume_file is None:
        return "❌ Please upload your resume first to begin the interview.", "", "", "00:00", gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Read resume content
        if hasattr(resume_file, 'name'):
            with open(resume_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                resume_content = f.read()
        else:
            resume_content = str(resume_file)
        
        # Analyze resume
        analysis = interview_bot.analyze_resume(resume_content)
        
        if 'personalized_questions' in analysis:
            interview_bot.personalized_questions = analysis['personalized_questions']
        
        # Start the timer
        interview_bot.start_timer()
        
        # Get first question
        first_question, should_continue = interview_bot.get_next_question()
        
        welcome_msg = f"""
# 🎯 Interview Started Successfully!

## 📊 Resume Analysis Complete
- **Skills Identified:** {', '.join(analysis.get('skills', ['General skills']))}
- **Experience Level:** {analysis.get('experience_years', 0)} years
- **Professional Roles:** {', '.join(analysis.get('roles', ['Professional']))}

## ⏰ Interview Timer: 15 Minutes Active

## 📋 First Question:
**{first_question}**

---
*Please provide your answer in the text box below. The interview will automatically end after 15 minutes.*
        """
        
        return (
            welcome_msg, 
            first_question, 
            "",  # Clear answer box
            interview_bot.get_time_remaining_formatted(),
            gr.update(visible=True),  # Show interview controls
            gr.update(visible=True)   # Show exit button
        )
        
    except Exception as e:
        return f"❌ Error processing resume: {str(e)}", "", "", "00:00", gr.update(visible=False), gr.update(visible=False)

def submit_answer(current_question, user_answer):
    """Process user's answer and get next question"""
    global interview_bot
    
    if not interview_bot.interview_data['is_active']:
        return generate_final_interview_report(), "", "", interview_bot.get_time_remaining_formatted()
    
    if not user_answer.strip():
        return "⚠️ Please provide an answer before submitting.", current_question, user_answer, interview_bot.get_time_remaining_formatted()
    
    # Evaluate the current answer
    evaluation = interview_bot.evaluate_answer(current_question, user_answer)
    
    # Get next question
    next_question, should_continue = interview_bot.get_next_question()
    
    if should_continue and interview_bot.interview_data['is_active']:
        # Continue interview
        response = f"""
## 📊 Answer Evaluation
- **Score:** {evaluation['score']}/100 
- **Feedback:** {evaluation['feedback']}

### ✅ Strengths
{chr(10).join(f"• {strength}" for strength in evaluation['strengths']) if evaluation['strengths'] else '• Keep building on your responses'}

### 🎯 Areas to Improve
{chr(10).join(f"• {improvement}" for improvement in evaluation['improvements']) if evaluation['improvements'] else '• Continue with your current approach'}

---

## 📋 Next Question:
**{next_question}**

*Time Remaining: {interview_bot.get_time_remaining_formatted()}*
        """
        
        return response, next_question, "", interview_bot.get_time_remaining_formatted()
    else:
        # End interview and generate report
        return generate_final_interview_report(), "", "", interview_bot.get_time_remaining_formatted()

def exit_interview():
    """Allow user to exit interview early"""
    global interview_bot
    interview_bot.force_end_interview("User requested exit")
    return generate_final_interview_report(), "", "", "00:00"

def generate_final_interview_report():
    """Generate and return the final interview report"""
    global interview_bot
    report = interview_bot.generate_final_report()
    
    if 'error' in report:
        return f"❌ {report['error']}"
    
    # Format the comprehensive report
    final_report = f"""
# 📊 Interview Performance Report

## 🎯 Overall Performance
- **Final Score:** {report['interview_summary']['overall_score']}/100
- **Performance Level:** {report['interview_summary']['performance_level']}
- **Consistency Score:** {report['interview_summary']['consistency_score']}/100

> {report['interview_summary']['performance_description']}

## 📈 Session Summary
- **Questions Answered:** {report['session_details']['questions_answered']}
- **Interview Duration:** {report['session_details']['duration_minutes']} minutes
- **Completion Status:** {report['session_details']['completion_status']}
- **Score Range:** {report['session_details']['lowest_score']} - {report['session_details']['highest_score']}

## 💪 Key Strengths
{chr(10).join(f"✅ {strength}" for strength in report['key_insights']['strengths']) if report['key_insights']['strengths'] else '• Continue developing your interview skills'}

## 🎯 Areas for Improvement  
{chr(10).join(f"🔄 {improvement}" for improvement in report['key_insights']['areas_for_improvement']) if report['key_insights']['areas_for_improvement'] else '• Keep up the excellent work'}

## 📋 Detailed Question Breakdown
{chr(10).join(f"**Q{item['question_number']}:** {item['score']}/100 - {item['feedback'][:100]}..." for item in report['detailed_breakdown'])}

## 🚀 Personalized Recommendations
{chr(10).join(f"💡 {rec}" for rec in report['recommendations'])}

## 📊 Final Recommendation
> **{report['interview_summary']['recommendation']}**

---
*Interview completed on {report['metadata']['timestamp'][:10]} | AI Resume Interview Bot v{report['metadata']['interview_version']}*

🎉 **Thank you for using the AI Interview Bot! Best of luck with your job search!**
    """
    
    return final_report

def update_timer():
    """Update timer display (called periodically)"""
    global interview_bot
    if interview_bot.interview_data['is_active']:
        return interview_bot.get_time_remaining_formatted()
    return "00:00"

# Create Gradio interface with enhanced UI
with gr.Blocks(title="AI Resume Interview Bot - 15 Min Timed Session", theme=gr.themes.Soft()) as app:
    
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 2.5em; margin: 0;">🎯 AI Data Science Interview Bot</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 10px 0;">15-Minute Timed Data Science Interview Practice with AI-Powered Scoring & Feedback</p>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin-top: 15px;">
            <p style="color: white; margin: 0;"><strong>⏱️ Time Limit:</strong> 15 Minutes | <strong>📊 DS-Focused Scoring</strong> | <strong>🎯 Project-Based Questions</strong></p>
        </div>
    </div>
    """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=2):
            # Setup Section
            gr.HTML("<h3 style='color: #667eea;'>📄 Step 1: Upload Your Resume</h3>")
            resume_file = gr.File(
                label="📎 Upload Resume", 
                file_types=[".txt", ".pdf", ".doc", ".docx"],
                elem_id="resume_upload"
            )
            
            with gr.Row():
                start_btn = gr.Button(
                    "🚀 Start 15-Min Interview", 
                    variant="primary", 
                    size="lg",
                    elem_id="start_button"
                )
                
        with gr.Column(scale=3):
            # Timer and Status
            with gr.Row():
                timer_display = gr.Textbox(
                    label="⏰ Time Remaining",
                    value="15:00",
                    interactive=False,
                    elem_id="timer"
                )
                
                exit_btn = gr.Button(
                    "🚪 Exit Interview",
                    variant="stop",
                    visible=False,
                    elem_id="exit_button"
                )
            
            # Interview Section  
            gr.HTML("<h3 style='color: #667eea;'>💬 Step 2: Answer Interview Questions</h3>")
            interview_output = gr.Markdown(
                value="📋 Upload your resume above to begin your 15-minute interview session.",
                elem_id="interview_display"
            )
    
    # Answer Input Section
    with gr.Group(visible=False, elem_id="answer_section") as answer_section:
        current_question = gr.Textbox(
            label="📋 Current Question",
            value="",
            interactive=False,
            lines=2,
            elem_id="current_question"
        )
        
        user_answer = gr.Textbox(
            label="✍️ Your Answer",
            placeholder="Provide a detailed answer here (aim for 50-120 words for optimal scoring)...",
            lines=4,
            elem_id="user_answer"
        )
        
        with gr.Row():
            submit_btn = gr.Button(
                "✅ Submit Answer & Continue",
                variant="primary",
                size="lg"
            )
            
            skip_btn = gr.Button(
                "⏭️ Skip Question",
                variant="secondary"
            )
    
    # Instructions and Tips
    gr.HTML("""
    <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; margin-top: 20px; border-left: 4px solid #667eea;">
        <h4 style="color: #667eea; margin-top: 0;">💡 Data Science Interview Tips for Best Scores:</h4>
        <ul style="color: #555;">
            <li><strong>Technical Details:</strong> Mention specific algorithms, metrics (accuracy, precision, recall, F1-score), and tools used</li>
            <li><strong>Project Structure:</strong> Follow data science workflow - problem definition, EDA, modeling, validation, deployment</li>
            <li><strong>Business Impact:</strong> Quantify results with numbers, percentages, and measurable outcomes</li>
            <li><strong>Technical Depth:</strong> Explain your methodology, feature engineering, model selection rationale</li>
            <li><strong>End-to-End Thinking:</strong> Cover data collection, preprocessing, modeling, evaluation, and production deployment</li>
            <li><strong>Communication:</strong> Balance technical accuracy with clear explanations for diverse audiences</li>
        </ul>
        <div style="background: #e8f4ff; padding: 15px; margin-top: 15px; border-radius: 8px;">
            <strong>📈 Common DS Interview Topics:</strong> Projects, ML/DL Concepts, Statistical Foundations, Data Processing, Model Deployment, Business Impact
        </div>
    </div>
    """)
    
    # Event Handlers
    start_btn.click(
        start_interview,
        inputs=[resume_file],
        outputs=[interview_output, current_question, user_answer, timer_display, answer_section, exit_btn]
    )
    
    submit_btn.click(
        submit_answer,
        inputs=[current_question, user_answer],
        outputs=[interview_output, current_question, user_answer, timer_display]
    )
    
    skip_btn.click(
        submit_answer,
        inputs=[current_question, gr.Textbox(value="Question skipped by user.")],
        outputs=[interview_output, current_question, user_answer, timer_display]
    )
    
    exit_btn.click(
        exit_interview,
        outputs=[interview_output, current_question, user_answer, timer_display]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 30px;">
        <p><strong>🤖 AI Data Science Interview Bot</strong> | Powered by Hugging Face Transformers</p>
        <p style="font-size: 0.9em;">Specialized for Data Science roles - Practice ML concepts, project discussions, and technical communication! 📊🚀</p>
    </div>
    """)

# Custom CSS for enhanced styling
app.css = """
#timer {
    font-size: 1.5em !important;
    font-weight: bold !important;
    text-align: center !important;
    color: #e74c3c !important;
}

#start_button {
    font-size: 1.2em !important;
    padding: 15px 30px !important;
}

#exit_button {
    background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
    color: white !important;
}

#interview_display {
    min-height: 400px !important;
    max-height: 600px !important;
    overflow-y: auto !important;
}

#current_question {
    background: #f0f4ff !important;
    border-left: 4px solid #667eea !important;
}

#user_answer {
    border-left: 4px solid #27ae60 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
"""

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=True,
        show_error=True
    )