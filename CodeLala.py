import gradio as gr
import pandas as pd
import os
import re
import tempfile
import PyPDF2
from datetime import datetime
from openai import OpenAI

# Initialize OpenAI client for Gemini API
model = OpenAI(api_key="GeminiKey", 
               base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def get_gemini_response(prompt):
    """Function to get response from Gemini API using the existing my_googler function pattern"""
    messages = [
        {'role': "system", "content": "You are an expert study planner for students preparing for exams. if students provide anything else than syllabus or unrealted to the enginnering subjects topics, you will not be able to help them. just say 'I can only help with syllabus. just say please upload proper syllabus. '"},
        {'role': "user", "content": prompt},
    ]
    response = model.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages
    )
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF syllabus"""
    if pdf_file is None:
        return None
    
    text = ""
    try:
        # Get the file path from the uploaded file
        file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        
        # Open and read the PDF directly
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        
        # Clean and truncate text if too long
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 12000:  # Truncate if too long for the API
            text = text[:12000] + "... [truncated]"
            
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_file(file):
    """Extract text from uploaded file (PDF, TXT, DOCX)"""
    if file is None:
        return None
    
    text = ""
    try:
        file_path = file.name if hasattr(file, 'name') else file
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # Process PDF
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
        
        elif file_extension == '.txt':
            # Process TXT
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        elif file_extension == '.docx':
            # For DOCX files, you'd need python-docx library
            # If not available, provide a helpful message
            text = "DOCX file detected. Please install python-docx library for full DOCX support."
        
        # Clean and truncate text if too long
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 12000:  # Truncate if too long for the API
            text = text[:12000] + "... [truncated]"
            
        return text
    
    except Exception as e:
        return f"Error extracting text from file: {str(e)}"

def analyze_syllabus(syllabus_text, subject):
    """Analyze syllabus to identify key topics"""
    if not syllabus_text:
        return None
    
    prompt = f"""
    You are an expert educational consultant analyzing a course syllabus.
    
    SYLLABUS CONTENT:
    {syllabus_text}
    
    Based on this syllabus for {subject}, please:
    
    1. Identify and list all the major topics covered
    2. Highlight the top 20% most important topics that likely cover 80% of exam content (Pareto principle)
    3. For each high-priority topic, explain briefly why it's important (e.g., fundamental concept, frequently tested, etc.)
    
    Format your response as a structured JSON with these sections:
    - all_topics: [list of all topics]
    - high_priority_topics: [list of the 20% most important topics]
    - topic_importance: {{"topic1": "reason for importance", "topic2": "reason for importance", ...}}
    
    Use your educational expertise to identify truly high-yield topics.
    """
    
    response = get_gemini_response(prompt)
    return response

def generate_study_plan(subject, days_left, hours_per_day, resource_type, feedback_preference, syllabus_file=None):
    """Generate a personalized study plan based on user inputs and optional syllabus"""
    
    # Process syllabus if provided
    syllabus_text = None
    syllabus_analysis = None
    
    if syllabus_file is not None:
        syllabus_text = extract_text_from_pdf(syllabus_file)
        if syllabus_text and len(syllabus_text) > 100:  # Only analyze if we got meaningful text
            syllabus_analysis = analyze_syllabus(syllabus_text, subject)
    
    # Construct a detailed prompt for the Gemini model
    prompt = f"""
    As an expert educational assistant, create a personalized last-minute study plan with the following details:
    
    STUDENT SITUATION:
    - Subject: {subject}
    - Days remaining until exam: {days_left}
    - Available study hours per day: {hours_per_day}
    - Primary study resource: {resource_type}
    - Learning preference: {feedback_preference}
    """
    
    # Add syllabus analysis if available
    if syllabus_analysis:
        prompt += f"""
    SYLLABUS ANALYSIS:
    {syllabus_analysis}
    
    Base your study plan primarily on the high-priority topics identified in the syllabus analysis.
    """
    
    # Complete the prompt
    prompt += """
    Please provide a comprehensive study plan with:
    1. A day-by-day breakdown showing exactly which topics to cover each day
    2. Priority ranking of the most high-yield topics (top 20% that will likely cover 80% of exam content)
    3. For each major topic, suggest 2-3 specific prompts the student can use to ask ChatGPT/Gemini for deeper understanding
    4. Suggest 5-minute breaks and how to utilize them effectively between study sessions
    5. A brief motivational message for the student
    
    Format your response with clear headings, bullet points, and a visually organized structure.
    """
    
    # Get response from Gemini
    response = get_gemini_response(prompt)
    
    # Log the generation for feedback
    log_interaction(subject, days_left, hours_per_day, resource_type, feedback_preference)
    
    return response

def generate_practice_questions(subject, topic, materials_file=None):
    """Generate practice questions for a specific topic using optional topic materials"""
    
    # Check if the topic is study-related
    study_related_keywords = [
        # Computer Science/IT
        "algorithm", "data structure", "programming", "software", "database", 
        "operating system", "network", "machine learning", "artificial intelligence",
        "web", "development", "computation", "architecture", "compiler", "memory",
        "process", "thread", "sql", "query", "normalization", "index", "transaction",
        "protocol", "routing", "encryption", "security", "api", "interface",
        
        # Mathematics
        "calculus", "algebra", "geometry", "statistics", "probability", "theorem",
        "equation", "function", "matrix", "vector", "differential", "integral",
      
        # General academic
        "theory", "concept", "principle", "law", "formula", "method", "analysis",
        "design", "evaluation", "research", "study", "experiment", "model"
    ]
    
    # Predefined subjects (hardcoded to avoid scope issues)
    predefined_subjects = [
        "data structures & algorithms", "dsa",
        "operating systems", "os",
        "database management systems", "dbms",
        "computer networks", "cn",
        "machine learning", "ml",
        "web development",
        "software engineering",
        "artificial intelligence",
        "theory of computation",
        "computer architecture"
    ]
    
    # Check if topic contains any study-related keywords
    is_study_related = any(keyword.lower() in topic.lower() or keyword.lower() in subject.lower() 
                          for keyword in study_related_keywords)
    
    # Also check if subject is one of our predefined subjects
    is_predefined_subject = any(sub.lower() in subject.lower() for sub in predefined_subjects)
    
    if not (is_study_related or is_predefined_subject):
        return "I can only generate practice questions for academic or study-related topics. Please enter a topic related to your studies or coursework."
    
    # Process materials if provided
    materials_text = None
    
    if materials_file is not None:
        materials_text = extract_text_from_file(materials_file)
    
    # Base prompt
    prompt = f"""
    Create 5 high-quality practice questions for the topic '{topic}' in the subject '{subject}'. 
    
    For each question:
    1. Start with a challenging but fair question that tests deep understanding
    2. Provide a detailed solution
    3. Add a brief explanation of the key concept being tested
    
    Format each question clearly with numbers and visual separation.
    """
    
    # Add materials content if available
    if materials_text and len(materials_text) > 100:
        prompt += f"""
    
    BASE YOUR QUESTIONS ON THE FOLLOWING MATERIALS:
    {materials_text}
    
    Make sure the questions are directly relevant to the content in these materials, 
    focusing on the key concepts, formulas, and techniques mentioned.
    """
    else:
        # If no materials provided, emphasize focusing on the standard curriculum
        prompt += """
        
    Focus on standard curriculum content for this topic that would typically appear in exams.
    Cover different aspects and difficulty levels of this topic.
    """
    
    return get_gemini_response(prompt)

def generate_smart_prompts(subject, topic):
    """Generate smart prompts to ask AI about a specific topic"""
    
    prompt = f"""
    Generate 5 effective prompts that a student can use to ask ChatGPT/Gemini about the topic '{topic}' in '{subject}'.
    
    For each prompt:
    1. Make it specific and focused on a particular aspect of the topic
    2. Design it to extract conceptual understanding rather than just facts
    3. Frame it to get explanations with analogies or visualizations
    4. Add a brief note on what kind of insight this prompt is designed to extract
    
    Format as a numbered list with clear separation between prompts.
    """
    
    return get_gemini_response(prompt)

def log_interaction(subject, days_left, hours_per_day, resource_type, feedback_preference):
    """Log user interactions to a CSV file for future improvements"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "user_interactions.csv")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("timestamp,subject,days_left,hours_per_day,resource_type,feedback_preference\n")
    
    # Append the new interaction
    with open(log_file, 'a') as f:
        f.write(f"{timestamp},{subject},{days_left},{hours_per_day},{resource_type},{feedback_preference}\n")
    
    return True

def save_feedback(feedback_type, feedback_text, plan_details):
    """Save user feedback for continuous improvement"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    feedback_dir = "feedback"
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    
    feedback_file = os.path.join(feedback_dir, "user_feedback.csv")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            f.write("timestamp,feedback_type,feedback_text,plan_details\n")
    
    # Clean the plan details to avoid CSV formatting issues
    plan_details_clean = str(plan_details).replace(",", ";").replace("\n", " ")
    feedback_text_clean = str(feedback_text).replace(",", ";").replace("\n", " ")
    
    # Append the new feedback
    with open(feedback_file, 'a') as f:
        f.write(f"{timestamp},{feedback_type},{feedback_text_clean},{plan_details_clean}\n")
    
    return "Thank you for your feedback! It helps us improve future study plans."

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Define CSS for styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    
    h1 {
        color: #2E5090;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2E5090;
        font-size: 1.8rem;
        margin-top: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    
    .footer {
        text-align: center;
        color: #777;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    
    .output-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .tab-content {
        padding: 20px;
    }
    
    .syllabus-upload {
        border: 2px dashed #2E5090;
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f7ff;
        margin: 15px 0;
        transition: all 0.3s;
    }
    
    .syllabus-upload:hover {
        background-color: #e6f0ff;
        border-color: #1a3a70;
    }
    
    .syllabus-label {
        color: #2E5090;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .syllabus-help {
        color: #666;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    .feature-badge {
        background-color: #2E5090;
        color: white;
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 10px;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    .upload-icon {
        font-size: 1.5rem;
        color: #2E5090;
        margin-right: 10px;
    }
    
    .other-subject-field {
        margin-top: 10px;
        padding-left: 15px;
        border-left: 3px solid #2E5090;
    }
    """
    
    # Subjects list
    subjects = [
        "Data Structures & Algorithms (DSA)",
        "Operating Systems (OS)",
        "Database Management Systems (DBMS)",
        "Computer Networks (CN)",
        "Machine Learning (ML)",
        "Web Development",
        "Software Engineering",
        "Artificial Intelligence",
        "Theory of Computation",
        "Computer Architecture",
        "Other (specify below)"
    ]
    
    # Resource types
    resource_types = [
        "Textbooks",
        "Video lectures",
        "Online course materials",
        "Class notes",
        "Practice problems",
        "Previous year papers",
        "Mix of multiple resources"
    ]
    
    # Learning preferences
    learning_preferences = [
        "Visual learner (diagrams, charts)",
        "Auditory learner (discussions, explanations)",
        "Reading/writing learner (notes, summaries)",
        "Kinesthetic learner (practice, examples)",
        "Mix of multiple styles"
    ]
    
    # Create main tab for generating study plan
    with gr.Blocks(css=css) as app:
        gr.HTML("<h1>😄 CodeLala</h1>")
        gr.HTML("<div class='subtitle'>Your AI-powered exam preparation assistant</div>")
        
        with gr.Tab("Generate Study Plan"):
            with gr.Row():
                with gr.Column(scale=1):
                    subject = gr.Dropdown(
                        subjects, 
                        label="Select Subject", 
                        info="Choose the subject you need to prepare for"
                    )
                    
                    # Add conditional textbox for "Other" subject
                    other_subject = gr.Textbox(
                        label="Specify Other Subject",
                        placeholder="Enter your specific subject name...",
                        visible=False,
                        elem_classes=["other-subject-field"]
                    )
                    
                    # Make textbox visible when "Other" is selected
                    def toggle_other_subject(choice):
                        return {"visible": choice == "Other (specify below)"}
                    
                    subject.change(toggle_other_subject, inputs=[subject], outputs=[other_subject])
                    
                    days_left = gr.Slider(
                        minimum=1, 
                        maximum=15, 
                        value=3, 
                        step=1, 
                        label="Days Left Until Exam",
                        info="How many days do you have before the exam?"
                    )
                    hours_per_day = gr.Slider(
                        minimum=1, 
                        maximum=12, 
                        value=4, 
                        step=1, 
                        label="Study Hours Per Day",
                        info="Realistic hours you can dedicate each day"
                    )
                    resource_type = gr.Dropdown(
                        resource_types, 
                        label="Primary Study Resource", 
                        info="What materials will you mainly use?"
                    )
                    feedback_preference = gr.Dropdown(
                        learning_preferences, 
                        label="Learning Preference", 
                        info="How do you learn best?"
                    )
                    
                    # Add syllabus upload feature
                    gr.HTML("<div class='syllabus-upload'>")
                    gr.HTML("<div class='syllabus-label'><span class='upload-icon'>📄</span> Upload Syllabus <span class='feature-badge'>NEW</span></div>")
                    syllabus_file = gr.File(
                        label="",
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    gr.HTML("<div class='syllabus-help'>Optional: Upload your course syllabus (PDF) for more targeted recommendations. Our AI will analyze your syllabus to identify the most important topics to focus on.</div>")
                    gr.HTML("</div>")
                    
                    generate_btn = gr.Button("Generate My Study Plan", variant="primary")
                
                with gr.Column(scale=2):
                    study_plan_output = gr.Markdown(
                        label="Your Personalized Study Plan",
                        value="Your study plan will appear here...",
                    )
            
            with gr.Row():
                with gr.Column():
                    feedback_type = gr.Radio(
                        ["Very Helpful", "Somewhat Helpful", "Not Helpful"], 
                        label="Was this plan helpful?",
                        info="Your feedback helps us improve"
                    )
                    feedback_text = gr.Textbox(
                        label="Additional Feedback (Optional)",
                        placeholder="Tell us how we can make your study plan better...",
                        lines=2
                    )
                    feedback_btn = gr.Button("Submit Feedback")
                    feedback_result = gr.Textbox(label="Feedback Status")
        
        with gr.Tab("Generate Practice Questions"):
            with gr.Row():
                with gr.Column(scale=1):
                    practice_subject = gr.Dropdown(
                        subjects, 
                        label="Select Subject", 
                        info="Choose the subject"
                    )
                    
                    # Add conditional textbox for "Other" subject
                    practice_other_subject = gr.Textbox(
                        label="Specify Other Subject",
                        placeholder="Enter your specific subject name...",
                        visible=False,
                        elem_classes=["other-subject-field"]
                    )
                    
                    # Make textbox visible when "Other" is selected
                    practice_subject.change(toggle_other_subject, inputs=[practice_subject], outputs=[practice_other_subject])
                    
                    practice_topic = gr.Textbox(
                        label="Specific Topic",
                        placeholder="Enter a specific topic (e.g., 'Binary Trees', 'Process Scheduling')",
                        info="Be specific for better questions"
                    )
                    
                    # Add materials upload feature
                    gr.HTML("<div class='syllabus-upload'>")
                    gr.HTML("<div class='syllabus-label'><span class='upload-icon'>📚</span> Upload Topic Materials <span class='feature-badge'>NEW</span></div>")
                    practice_materials = gr.File(
                        label="",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="single"
                    )
                    gr.HTML("<div class='syllabus-help'>Optional: Upload notes, textbook pages, or any material related to this topic for more focused practice questions.</div>")
                    gr.HTML("</div>")
                    
                    practice_btn = gr.Button("Generate Practice Questions", variant="primary")
        
                with gr.Column(scale=2):
                    practice_output = gr.Markdown(
                        label="Practice Questions",
                        value="Your practice questions will appear here..."
                    )
        
        with gr.Tab("Smart Prompt Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_subject = gr.Dropdown(
                        subjects, 
                        label="Select Subject", 
                        info="Choose the subject"
                    )
                    
                    # Add conditional textbox for "Other" subject
                    prompt_other_subject = gr.Textbox(
                        label="Specify Other Subject",
                        placeholder="Enter your specific subject name...",
                        visible=False,
                        elem_classes=["other-subject-field"]
                    )
                    
                    # Make textbox visible when "Other" is selected
                    prompt_subject.change(toggle_other_subject, inputs=[prompt_subject], outputs=[prompt_other_subject])
                    
                    prompt_topic = gr.Textbox(
                        label="Specific Topic",
                        placeholder="Enter a specific topic (e.g., 'Sorting Algorithms', 'Memory Management')",
                        info="Be specific for better prompts"
                    )
                    prompt_btn = gr.Button("Generate Smart Prompts", variant="primary")
                
                with gr.Column(scale=2):
                    prompt_output = gr.Markdown(
                        label="Smart Prompts for ChatGPT/Gemini",
                        value="Your smart prompts will appear here..."
                    )
        
        gr.HTML("<div class='footer'>CodeLala - Helping students ace their exams since 2025</div>")
        
        # Function to handle "Other" subject selection
        def get_final_subject(dropdown_value, other_value):
            if dropdown_value == "Other (specify below)" and other_value:
                return other_value
            return dropdown_value
        
        # Set up event handlers with the combined subject inputs
        def handle_study_plan(dropdown_subject, other_subject, days_left, hours_per_day, resource_type, feedback_preference, syllabus_file):
            final_subject = get_final_subject(dropdown_subject, other_subject)
            
            # Show a processing message if syllabus is uploaded
            if syllabus_file is not None:
                processing_message = "⏳ Analyzing your syllabus to identify key topics... This may take a moment."
                yield processing_message
            
            # Generate the actual study plan
            result = generate_study_plan(final_subject, days_left, hours_per_day, resource_type, feedback_preference, syllabus_file)
            
            # Return the final result
            yield result
        
        def handle_practice_questions(dropdown_subject, other_subject, topic, materials_file):
            """Handle practice question generation with subject selection and materials"""
            final_subject = get_final_subject(dropdown_subject, other_subject)
            
            # Validate inputs
            if not topic or topic.strip() == "":
                return "Please enter a specific topic to generate practice questions."
            
            # Show appropriate processing message
            if materials_file is not None:
                processing_message = "⏳ Analyzing your materials to create targeted practice questions... This may take a moment."
                yield processing_message
            else:
                processing_message = "⏳ Generating practice questions for your topic... This may take a moment."
                yield processing_message
            
            # Generate the practice questions
            result = generate_practice_questions(final_subject, topic, materials_file)
            
            # Return the final result
            yield result
        
        def handle_smart_prompts(dropdown_subject, other_subject, topic):
            final_subject = get_final_subject(dropdown_subject, other_subject)
            return generate_smart_prompts(final_subject, topic)
        
        # Connect the modified handlers to buttons
        generate_btn.click(
            handle_study_plan, 
            inputs=[subject, other_subject, days_left, hours_per_day, resource_type, feedback_preference, syllabus_file], 
            outputs=study_plan_output
        )
        
        feedback_btn.click(
            save_feedback, 
            inputs=[feedback_type, feedback_text, study_plan_output], 
            outputs=feedback_result
        )
        
        practice_btn.click(
            handle_practice_questions,
            inputs=[practice_subject, practice_other_subject, practice_topic, practice_materials],
            outputs=practice_output
        )
        
        prompt_btn.click(
            handle_smart_prompts,
            inputs=[prompt_subject, prompt_other_subject, prompt_topic],
            outputs=prompt_output
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()