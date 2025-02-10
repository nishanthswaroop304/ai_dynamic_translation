from flask import Flask, render_template, request, jsonify
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import the new OpenAI client interface
from openai import OpenAI
# Import Groq client
from groq import Groq

app = Flask(__name__)

# Initialize the clients
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq()  # Assumes your groq client configuration is handled internally or via env

@app.route("/")
def index():
    # Get BLOB_TIME from environment (default to 3000 ms if not set)
    blob_time = int(os.getenv("BLOB_TIME", 3000))
    return render_template("index.html", blob_time=blob_time)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Retrieve the source language from the form ("Translate from")
    source_language = request.form.get("source_language", "English")
    # Mapping from language names to ISO-639-1 codes.
    language_codes = {
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it",
        "Hindi": "hi",
        "Korean": "ko"
    }
    iso_language = language_codes.get(source_language, None)
    
    # If the source language is German, include a German prompt.
    prompt_text = "Transkribiere diesen deutschen Audioinhalt." if source_language == "German" else None

    audio_file = request.files["audio"]
    # Save the uploaded audio to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcription_result = client_openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=iso_language,
                prompt=prompt_text
            )
        transcription_text = transcription_result.text
    except Exception as e:
        transcription_text = f"Error during transcription: {e}"
    finally:
        os.remove(tmp_path)
    
    # Return the original transcription.
    return jsonify({"original": transcription_text})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    text = data.get("text", "")
    source_language = data.get("source_language", "English")
    target_language = data.get("target_language", "French")
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate translation assistant. "
                    "Translate the following text from {} to {}. "
                    "If the input text is already natural, correct English, output it unchanged. "
                    "Otherwise, translate with utmost precision, preserving punctuation and grammar. "
                    "Output only the translated text as a single concise sentence with no extra commentary or formatting. "
                    "If there is an error, output a concise error message in plain text."
                ).format(source_language, target_language)
            },
            {
                "role": "user",
                "content": f"Translate the following text from {source_language} to {target_language}: '{text}'"
            }
        ]
        translation_response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        translated_text = translation_response.choices[0].message.content.strip()
    except Exception as e:
        translated_text = f"Error during translation: {e}"
    return jsonify({"translation": translated_text})

if __name__ == "__main__":
    app.run(debug=True)
