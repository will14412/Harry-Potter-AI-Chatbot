import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import aiml
import speech_recognition as sr
import nltk
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf

################################
#       INITIAL LOADS
################################

# Load the previously saved model
loaded_model = load_model('wand_casting_classifier_tuned.h5')

root = tk.Tk()
root.withdraw()

# Define the FOL template
FOL_TEMPLATE = "{}({})"

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

################################
#      CNN IMAGE PROCESS
################################


class_names = ['alohamora', 'engorgio', 'expelliarmus', 'flipendo', 'lumos', 'nox', 'rictumsempra', 'wingardiumleviosa']
def preprocess_for_inference(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # Keep as RGB
    img = tf.image.resize(img, [32, 32])  # Resize to match model's expected input
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img

def predict_class(image_path, model):
    img = preprocess_for_inference(image_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Index of the max probability
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name
    #print("Predicted probabilities for each class:", predictions[0])
    print("I believe that spell pattern is: ", predicted_class_name)


################################
#          LOAD QA KB
################################

def load_qa_kb(filename):
    qa_pairs = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            qa_pairs.append(row)
    return qa_pairs

filename = 'qa_kb.csv'
qa_kb = load_qa_kb(filename)

################################
#        DATA PROCESS
################################

def preprocess_data(questions):
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(questions)
    
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    
    return count_vectorizer, tfidf_transformer, X_tfidf

questions = [pair[0] for pair in qa_kb]
#print("Loaded {} questions from Q/A KB.".format(len(questions)))

# Preprocess data
vectorizer, _, X_tfidf = preprocess_data(questions)  # Ignoring the second returned value

vocabulary = vectorizer.get_feature_names_out()
#print("Vocabulary:", vocabulary)
#print("TF-IDF matrix shape:", X_tfidf.shape)


################################
#        AIML AGENT
################################

# Initialise AIML agent
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

################################
#        QAKB COSINE
################################

def qa_kb_cosine(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X_tfidf)
    closest_index = similarities.argmax()
    closest_similarity = similarities.max()
    closest_question = questions[closest_index]

    if closest_similarity > 0.6:
        #print(closest_similarity)
        if closest_similarity < 0.7:
            print("Did you mean:", closest_question)
            confirm = input("You: ").lower()
            if confirm == 'no':
                print("Bot: Sorry, I don't know.")
            else:
                for pair in qa_kb:
                    if pair[0] == closest_question:
                        if "command" in pair[1]:
                            fol_statement = command_check(pair[1])
                            return fol_statement
                        else:
                            print(pair[1])
                        break
        else:
            for pair in qa_kb:
                if pair[0] == closest_question:
                    if "command" in pair[1]:
                        fol_statement = command_check(pair[1])
                        return fol_statement
                    else:
                        print(pair[1])
                    break
    else:
        print("Sorry, I'm unsure.")
        return user_input  # Return the user input when unsure or no command found

def command_check(answer):
        command_number = int(answer[len("command"):].rstrip('.'))
        if command_number == 1:
            user_input = recognize_speech()
            return qa_kb_cosine(user_input)
        elif command_number == 2:
            user_input = input("Of course, what would you like me to check:")
            fol_statements = fol_statements_lp(csv_file_path)
            fol_statement = convert_to_fol(user_input)

            check_fact(fol_statement,fol_statements)
        elif command_number == 3:
            user_input = input("Of course, what would you like me to learn:")
            fol_statements = fol_statements_lp(csv_file_path)
            teach_fact(user_input,fol_statements)
        elif command_number == 4:
            print("Please upload an image:")
            image_path = filedialog.askopenfilename()
            if image_path:
                print("Uploaded Image Path:", image_path)
                predict_class(image_path, loaded_model)
            else:
                print("No file selected.")
################################
#              FOL
################################

def convert_to_fol(user_input):
    # Tokenize the input
    tokens = word_tokenize(user_input)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalnum()]
    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in filtered_tokens]
    # Reverse the order of tokens
    reversed_tokens = lemmatized_tokens[::-1]
    # Remove spaces between words
    stripped_tokens = [token.replace(" ", "") for token in reversed_tokens]
    # Format tokens into FOL notation without comma
    fol_statement = FOL_TEMPLATE.format(stripped_tokens[0].capitalize(), ''.join(stripped_tokens[1:]))
    
    return fol_statement


csv_file_path = 'FOL_kb.csv'
def fol_statements_lp(filename):
    fol_statements = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  
                statement = row[0].strip()  # Remove leading/trailing whitespace
                fol_expression = Expression.fromstring(statement)
                fol_statements.append(fol_expression)
    return fol_statements



def write_to_fol_kb(fol_statement):
    with open('FOL_kb.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([fol_statement])
        
def teach_fact(user_input, fol_statements):
    fol_statement = convert_to_fol(user_input)
    write_to_fol_kb(fol_statement)
    print("The fact has been successfully added to the knowledge base.")

    
def check_for_contradictions(new_fol_statement, existing_fol_statements):
    # Negate the new FOL statement
    negated_new_fol_statement = f"~({new_fol_statement})"

    # Combine the existing FOL statements and the negated new statement
    combined_statements = existing_fol_statements + [negated_new_fol_statement]

    # Try to prove that the combined statements lead to a contradiction
    prover = ResolutionProver()
    result = prover.prove(Expression.fromstring('false'), combined_statements)

    # If result is True, it means a contradiction was found
    return result


read_expr = Expression.fromstring

def check_fact(fol_statement, fol_statements):
    # Assume fol_statements are already parsed expressions
    kb_exprs = fol_statements  # Directly use them without reparsing
    
    # Ensure fol_statement is in the correct format for processing
    if isinstance(fol_statement, str):
        fact_expr = read_expr(fol_statement)
    else:
        fact_expr = fol_statement  # Assuming fol_statement is already an expression if not a str
    
    # Use NLTK's resolution prover to check if the fact can be deduced from the KB
    if ResolutionProver().prove(fact_expr, kb_exprs, verbose=False):
        print("The fact is consistent with the knowledge base.")
    else:
        print("The fact contradicts the knowledge base.")

################################
#     SPEECH RECOGNITION
################################

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        print("You (voice):", user_input)
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand your audio.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

################################
#      USER INTERACTION
################################

# Welcome user

print("Welcome to iPotter, the Harry Potter chatbot, feel free to ask me anything!")

while True:
    input_mode = input("What do you want to know or do:").lower()
    cmd = 0
    user_input = input_mode  
    # Process the input (AIML, QA KB, etc.)
    aiml_response = kern.respond(user_input.upper())
    if aiml_response:
        print("AIML:", aiml_response)
    else:
        qa_kb_cosine(user_input)
        
        
        
        
        
        
        
        