import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import random
from gtts import gTTS
import os
import streamlit.components.v1 as components

# Load the DialoGPT model and tokenizer for conversational responses
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the translation model and tokenizer for English to Spanish translation
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Define an expanded vocabulary dictionary for simple translation exercises
vocabulary = {
    'apple': 'manzana',
    'book': 'libro',
    'cat': 'gato',
    'dog': 'perro',
    'house': 'casa',
    'tree': 'árbol',
    'water': 'agua',
    'sun': 'sol',
    'moon': 'luna',
    'star': 'estrella'
}

# Define expanded quiz questions and answers with difficulty levels
quiz_questions = [
    {'question': 'What is the Spanish word for apple?', 'answer': 'manzana', 'difficulty': 'easy'},
    {'question': 'What is the Spanish word for book?', 'answer': 'libro', 'difficulty': 'easy'},
    {'question': 'What is the Spanish word for cat?', 'answer': 'gato', 'difficulty': 'easy'},
    {'question': 'What is the Spanish word for dog?', 'answer': 'perro', 'difficulty': 'easy'},
    {'question': 'What is the Spanish word for house?', 'answer': 'casa', 'difficulty': 'medium'},
    {'question': 'What is the Spanish word for tree?', 'answer': 'árbol', 'difficulty': 'medium'},
    {'question': 'What is the Spanish word for water?', 'answer': 'agua', 'difficulty': 'medium'},
    {'question': 'What is the Spanish word for sun?', 'answer': 'sol', 'difficulty': 'hard'},
    {'question': 'What is the Spanish word for moon?', 'answer': 'luna', 'difficulty': 'hard'},
    {'question': 'What is the Spanish word for star?', 'answer': 'estrella', 'difficulty': 'hard'}
]

# Streamlit app layout
st.title("Language Learning Assistant")

st.sidebar.title("Menu")
options = ["Chat", "Translate", "Quiz"]
choice = st.sidebar.selectbox("Select an option", options)

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
        translation = vocabulary[text]
    else:
        translation = translate_text(text)
    return translation

def start_quiz(difficulty=None):
    if difficulty:
        filtered_questions = [q for q in quiz_questions if q['difficulty'] == difficulty]
        if not filtered_questions:
            return None, f"No questions available for difficulty '{difficulty}'. Try 'easy', 'medium', or 'hard'."
        question = random.choice(filtered_questions)
    else:
        question = random.choice(quiz_questions)
    
    return question, f"Here's your question: {question['question']}"

def translate_text(text):
    inputs = translation_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    translated_ids = translation_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def generate_conversational_response(message):
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
    difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"], key="quiz_difficulty")
    if st.button("Start Quiz"):
        question, quiz_message = start_quiz(difficulty)
        if question:
            st.text(quiz_message)
            choices = list(vocabulary.values())
            random.shuffle(choices)
            correct_choice = question['answer']
            if correct_choice not in choices:
                choices[random.randint(0, len(choices) - 1)] = correct_choice
            selected_choice = st.radio("Choose the correct answer:", choices, key="quiz_choices")
            if st.button("Submit Answer"):
                if selected_choice == correct_choice:
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect! The correct answer was '{correct_choice}'.")
        else:
            st.error(quiz_message)
