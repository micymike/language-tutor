import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import random
from gtts import gTTS
import os
import pandas as pd
from collections import defaultdict

# Vocabulary dictionary
vocabulary = {
    'apple': 'manzana', 'book': 'libro', 'cat': 'gato', 'dog': 'perro',
    'house': 'casa', 'tree': 'árbol', 'water': 'agua', 'sun': 'sol',
    'moon': 'luna', 'star': 'estrella', 'car': 'coche', 'computer': 'ordenador',
    'phone': 'teléfono', 'food': 'comida', 'friend': 'amigo', 'school': 'escuela',
    'city': 'ciudad', 'music': 'música', 'time': 'tiempo', 'family': 'familia'
}

# Quiz questions
quiz_questions = [
    {'question': 'What is the Spanish word for apple?', 'answer': 'manzana', 'difficulty': 'easy', 'category': 'Food'},
    {'question': 'What is the Spanish word for book?', 'answer': 'libro', 'difficulty': 'easy', 'category': 'Objects'},
    {'question': 'What is the Spanish word for cat?', 'answer': 'gato', 'difficulty': 'easy', 'category': 'Animals'},
    {'question': 'What is the Spanish word for dog?', 'answer': 'perro', 'difficulty': 'easy', 'category': 'Animals'},
    {'question': 'What is the Spanish word for house?', 'answer': 'casa', 'difficulty': 'medium', 'category': 'Places'},
    {'question': 'What is the Spanish word for tree?', 'answer': 'árbol', 'difficulty': 'medium', 'category': 'Nature'},
    {'question': 'What is the Spanish word for water?', 'answer': 'agua', 'difficulty': 'medium', 'category': 'Nature'},
    {'question': 'What is the Spanish word for sun?', 'answer': 'sol', 'difficulty': 'hard', 'category': 'Nature'},
    {'question': 'What is the Spanish word for moon?', 'answer': 'luna', 'difficulty': 'hard', 'category': 'Nature'},
    {'question': 'What is the Spanish word for star?', 'answer': 'estrella', 'difficulty': 'hard', 'category': 'Nature'},
    {'question': 'What is the Spanish word for car?', 'answer': 'coche', 'difficulty': 'medium', 'category': 'Transportation'},
    {'question': 'What is the Spanish word for computer?', 'answer': 'ordenador', 'difficulty': 'hard', 'category': 'Technology'},
    {'question': 'What is the Spanish word for phone?', 'answer': 'teléfono', 'difficulty': 'medium', 'category': 'Technology'},
    {'question': 'What is the Spanish word for food?', 'answer': 'comida', 'difficulty': 'easy', 'category': 'Food'},
    {'question': 'What is the Spanish word for friend?', 'answer': 'amigo', 'difficulty': 'easy', 'category': 'People'},
    {'question': 'What is the Spanish word for school?', 'answer': 'escuela', 'difficulty': 'medium', 'category': 'Places'},
    {'question': 'What is the Spanish word for city?', 'answer': 'ciudad', 'difficulty': 'medium', 'category': 'Places'},
    {'question': 'What is the Spanish word for music?', 'answer': 'música', 'difficulty': 'easy', 'category': 'Arts'},
    {'question': 'What is the Spanish word for time?', 'answer': 'tiempo', 'difficulty': 'hard', 'category': 'Abstract'},
    {'question': 'What is the Spanish word for family?', 'answer': 'familia', 'difficulty': 'easy', 'category': 'People'}
]

# Lazy loading for models
@st.cache_resource
def load_dialogue_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_translation_model():
    translation_model_name = "Helsinki-NLP/opus-mt-en-es"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)
    return translation_tokenizer, translation_model

# Helper functions
def generate_response(message):
    message = message.lower().strip()
    
    if message.startswith('translate '):
        return handle_translation(message[10:])
    elif message == 'quiz me':
        return start_quiz()
    elif message.startswith('quiz me '):
        return start_quiz(message.split()[-1])
    else:
        return generate_conversational_response(message)

def handle_translation(text):
    if text in vocabulary:
        return vocabulary[text]
    else:
        return translate_text(text)

def start_quiz(difficulty=None, category=None):
    filtered_questions = quiz_questions
    if difficulty:
        filtered_questions = [q for q in filtered_questions if q['difficulty'] == difficulty]
    if category:
        filtered_questions = [q for q in filtered_questions if q['category'] == category]
    
    if not filtered_questions:
        return None, f"No questions available for the selected criteria. Please try different options."
    
    question = random.choice(filtered_questions)
    return question, f"Here's your question: {question['question']}"

def translate_text(text):
    translation_tokenizer, translation_model = load_translation_model()
    inputs = translation_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    translated_ids = translation_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def generate_conversational_response(message):
    tokenizer, model = load_dialogue_model()
    inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt', max_length=512, truncation=True)
    response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def play_translation(translation):
    tts = gTTS(translation, lang='es')
    tts.save("translation.mp3")
    audio_file = open("translation.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    os.remove("translation.mp3")  # Clean up the file after playing

def spaced_repetition_quiz():
    if 'sr_questions' not in st.session_state:
        st.session_state.sr_questions = quiz_questions.copy()
        random.shuffle(st.session_state.sr_questions)
    
    if not st.session_state.sr_questions:
        st.session_state.sr_questions = quiz_questions.copy()
        random.shuffle(st.session_state.sr_questions)
    
    question = st.session_state.sr_questions.pop(0)
    return question, f"Here's your spaced repetition question: {question['question']}"

def update_progress(correct):
    if 'progress' not in st.session_state:
        st.session_state.progress = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    category = st.session_state.current_question['category']
    st.session_state.progress[category]['total'] += 1
    if correct:
        st.session_state.progress[category]['correct'] += 1

# Streamlit app layout
st.title("Optimized Language Learning Assistant")

st.sidebar.title("Menu")
options = ["Chat", "Translate", "Quiz", "Vocabulary", "Progress"]
choice = st.sidebar.selectbox("Select an option", options)

if choice == "Chat":
    st.header("Chat with the Bot")
    user_message = st.text_input("You:", key="chat_input")
    if st.button("Send"):
        if user_message:
            bot_response = generate_response(user_message)
            st.text_area("Bot:", bot_response, height=200)

elif choice == "Translate":
    st.header("Translate English to Spanish")
    text_to_translate = st.text_input("Enter text to translate:", key="translate_input")
    if st.button("Translate"):
        if text_to_translate:
            translation = handle_translation(text_to_translate)
            st.text_area("Translation:", translation, height=200)
            play_translation(translation)

elif choice == "Quiz":
    st.header("Quiz")
    quiz_type = st.radio("Select quiz type:", ["Regular", "Spaced Repetition"])
    
    if quiz_type == "Regular":
        difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard", "all"], key="quiz_difficulty")
        category = st.selectbox("Select category:", ["All"] + list(set(q['category'] for q in quiz_questions)), key="quiz_category")
        
        if 'current_question' not in st.session_state:
            st.session_state.current_question = None

        if st.button("Start Quiz") or st.session_state.current_question:
            difficulty = None if difficulty == "all" else difficulty
            category = None if category == "All" else category
            
            if not st.session_state.current_question:
                question, quiz_message = start_quiz(difficulty, category)
                if question:
                    st.session_state.current_question = question
                    st.session_state.quiz_message = quiz_message
                    st.session_state.user_answer = None
                else:
                    st.error(quiz_message)
                    st.session_state.current_question = None
            
            if st.session_state.current_question:
                st.text(st.session_state.quiz_message)
                choices = list(vocabulary.values())
                random.shuffle(choices)
                correct_choice = st.session_state.current_question['answer']
                if correct_choice not in choices:
                    choices[random.randint(0, len(choices) - 1)] = correct_choice
                selected_choice = st.radio("Choose the correct answer:", choices, key="quiz_choices")
                
                if st.button("Submit Answer"):
                    if selected_choice == correct_choice:
                        st.success("Correct!")
                        update_progress(True)
                    else:
                        st.error(f"Incorrect! The correct answer was '{correct_choice}'.")
                        update_progress(False)
                    
                    st.session_state.current_question = None  # Reset for next question
                    st.button("Next Question")

    elif quiz_type == "Spaced Repetition":
        if 'sr_current_question' not in st.session_state:
            st.session_state.sr_current_question = None

        if st.button("Start Spaced Repetition Quiz") or st.session_state.sr_current_question:
            if not st.session_state.sr_current_question:
                question, quiz_message = spaced_repetition_quiz()
                st.session_state.sr_current_question = question
                st.session_state.sr_quiz_message = quiz_message
                st.session_state.sr_user_answer = None

            st.text(st.session_state.sr_quiz_message)
            user_answer = st.text_input("Your answer:")
            
            if st.button("Submit Answer"):
                if user_answer.lower() == st.session_state.sr_current_question['answer'].lower():
                    st.success("Correct!")
                    update_progress(True)
                else:
                    st.error(f"Incorrect! The correct answer was '{st.session_state.sr_current_question['answer']}'.")
                    update_progress(False)
                
                st.session_state.sr_current_question = None  # Reset for next question
                st.button("Next Question")

elif choice == "Vocabulary":
    st.header("Vocabulary List")
    df = pd.DataFrame(list(vocabulary.items()), columns=['English', 'Spanish'])
    st.dataframe(df)

elif choice == "Progress":
    st.header("Your Progress")
    if 'progress' in st.session_state:
        progress_data = []
        for category, data in st.session_state.progress.items():
            correct = data['correct']
            total = data['total']
            percentage = (correct / total) * 100 if total > 0 else 0
            progress_data.append({'Category': category, 'Correct': correct, 'Total': total, 'Percentage': f"{percentage:.2f}%"})
        
        progress_df = pd.DataFrame(progress_data)
        st.dataframe(progress_df)
    else:
        st.info("No progress data available yet. Start taking quizzes to see your progress!")