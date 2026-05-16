from flask import Flask, render_template, request, jsonify
import os
import json
import re
import requests
import pdfplumber

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def analyze_resume(job_description, resume_text, candidate_name, groq_key):
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert HR recruiter. Always respond with valid JSON only. No markdown, no extra text."
            },
            {
                "role": "user",
                "content": f"""Analyze this resume against the job description.

JOB DESCRIPTION:
{job_description[:1000]}

RESUME ({candidate_name}):
{resume_text[:1500]}

Respond with ONLY this JSON:
{{
  "score": <number 0-100>,
  "match_level": "<Excellent|Good|Average|Poor>",
  "top_strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "gaps": ["<gap1>", "<gap2>"],
  "summary": "<2 sentence summary of fit>"
}}"""
            }
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    generated = result["choices"][0]["message"]["content"]
    json_match = re.search(r'\{.*\}', generated, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError("Could not parse response")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    groq_key = request.form.get('groq_key', '').strip()
    job_description = request.form.get('job_description', '').strip()
    resume_files = request.files.getlist('resumes')

    if not groq_key:
        return jsonify({'error': 'Please enter your Groq API key.'}), 400
    if not job_description:
        return jsonify({'error': 'Please enter a job description.'}), 400
    if not resume_files or all(f.filename == '' for f in resume_files):
        return jsonify({'error': 'Please upload at least one resume PDF.'}), 400

    results = []
    for resume_file in resume_files:
        if resume_file.filename == '':
            continue
        if not resume_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': f'"{resume_file.filename}" is not a PDF. Please upload PDF files only.'}), 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(save_path)

        try:
            resume_text = extract_text_from_pdf(save_path)
            if not resume_text:
                return jsonify({'error': f'Could not read text from "{resume_file.filename}". Make sure it is not a scanned image.'}), 400

            candidate_name = os.path.splitext(resume_file.filename)[0].replace('_', ' ').replace('-', ' ')
            analysis = analyze_resume(job_description, resume_text, candidate_name, groq_key)
            analysis['candidate'] = candidate_name
            results.append(analysis)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return jsonify({'error': 'Invalid Groq API key. Please check and try again.'}), 400
            return jsonify({'error': f'API error: {str(e)}'}), 500
        except Exception as e:
            return jsonify({'error': f'Something went wrong: {str(e)}'}), 500
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
