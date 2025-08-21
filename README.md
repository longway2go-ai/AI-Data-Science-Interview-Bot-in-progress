# 🤖 AI MCQ Based Quiz Agent

An intelligent interview preparation platform that generates personalized technical questions based on your resume using AI. Practice with multiple-choice questions tailored to your skills and experience level.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

## ✨ Features

- **📄 Resume Analysis**: Upload PDF/TXT resume for automatic skill extraction
- **🎯 Personalized Questions**: AI generates questions based on your resume content
- **📊 Multiple Difficulty Levels**: Easy, Moderate, and Hard question modes
- **⏱️ Timed Interviews**: 10-minute timer with visual progress tracking
- **🎲 Dynamic Question Generation**: Questions generated on-demand for faster loading
- **📈 Real-time Scoring**: Instant feedback with explanations
- **🧠 Dual AI Support**: Choose between Google Gemini or Groq (LLaMA) models
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🔄 Continuous Practice**: Retake interviews with different questions

## 🚀 Live Demo

[**Try the live application**](https://ai-interview-mcq-based-agent.onrender.com/)

## 🛠️ Technologies Used

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI Models**: Google Gemini Pro, Groq LLaMA, OpenAI
- **File Processing**: PyPDF2 for PDF parsing
- **Deployment**: Render (Free Tier)
- **UI Framework**: Custom CSS with Font Awesome icons

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API key **OR** Groq API key **OR** OpenAI
- Git (for deployment)

### 🔑 Getting API Keys

**Google Gemini API:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy your API key

**Groq API:**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up/Sign in
3. Navigate to API Keys section
4. Create new API key

**OpenAI API:**
1. Visit [OpenAI](https://makersuite.google.com/app/apikey)
2. Sign in/ Sign up
3. Click "Create API Key"
4. Copy your API key

## 🏠 Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/longway2go-ai/ai-interview-agent.git
   cd ai-interview-agent
   ```
2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create templates directory**
   ```bash
   mkdir templates
   ```
5. **Run the application**
   ```bash
   python app.py
   ```

## 🌐 Deployment on Render

### Quick Deploy
1. Fork this repository
2. Create account on [Render](https://render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Instance Type**: Free

### Manual Setup
Follow the detailed [deployment guide](https://render.com/docs/deploy-fastapi).

## 📁 Project Structure

```
ai-interview-agent/
├── app.py # Main FastAPI application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── templates/
│ └── index.html # Frontend interface
├── static/ # Static files (if any)
└── .gitignore # Git ignore rules
```


## 🎮 How to Use

1. **Upload Resume**: Select a PDF or TXT file containing your resume
2. **Choose AI Backend**: Select either Gemini or Groq
3. **Set Difficulty**: Pick Easy, Moderate, or Hard level
4. **Enter API Key**: Provide your chosen AI service API key
5. **Start Interview**: Begin the 30-question timed interview
6. **Answer Questions**: Select from multiple choice options
7. **Review Results**: Get detailed feedback and performance breakdown

## 🔧 Configuration

### Environment Variables (Optional)
```
PORT=8000 # Server port (auto-set by Render)
PYTHON_VERSION=3.11 # Python version for deployment
```

### Customization Options
- **Question Count**: Modify `total_questions = 30` in `app.py`
- **Timer Duration**: Change `timeRemaining = 600` (seconds) in JavaScript
- **Difficulty Levels**: Adjust prompt templates in `generate_single_mcq_question()`

## 📊 Features in Detail

### AI-Powered Question Generation
- Extracts key topics from your resume automatically
- Generates contextual questions based on your skills
- Adapts difficulty based on your experience level
- Provides detailed explanations for each answer

### Smart Resume Analysis
- Supports PDF and TXT formats
- Identifies technical skills, frameworks, and tools
- Focuses on relevant interview topics
- Handles various resume formats and layouts

### Interactive Interface
- Clean, modern design with smooth animations
- Real-time progress tracking
- Instant feedback with color-coded results
- Mobile-responsive layout

## 🐛 Troubleshooting

### Common Issues

**"API key not configured"**
- Ensure you've entered a valid API key
- Check that the key corresponds to the selected backend

**"Could not extract topics from resume"**
- Try a different resume format (PDF recommended)
- Ensure resume contains technical content
- Check that file size is reasonable (< 10MB)

**Questions loading slowly**
- This is normal for the first question generation
- AI model response time varies (typically 2-5 seconds)
- Consider switching to a different AI backend if persistent

**App not loading on Render**
- Check build logs in Render dashboard
- Verify all files are pushed to GitHub
- Ensure requirements.txt is in root directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for Python
- [Google Gemini](https://ai.google.dev/) - Powerful AI language model
- [Groq](https://groq.com/) - Ultra-fast AI inference
- [Render](https://render.com/) - Free hosting platform
- [Font Awesome](https://fontawesome.com/) - Beautiful icons

## 🔮 Future Enhancements

- [ ] Voice-based question reading (TTS integration)
- [ ] Speech-to-text answer input
- [ ] Interview performance analytics
- [ ] Custom question templates
- [ ] Team/company-specific question sets
- [ ] Export results to PDF
- [ ] Multi-language support

---

⭐ **Star this repository if you found it helpful!**

🐛 **Found a bug?** [Report it here](https://github.com/longway2go-ai/ai-interview-agent/issues)

📧 **Questions?** Feel free to reach out!
