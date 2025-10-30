import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main page with the file upload form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles file upload, performs analysis, and renders results."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    uploaded_file = request.files['file']

    if uploaded_file.filename == '' or not uploaded_file.filename.endswith('.csv'):
        return redirect(request.url)

    try:
        df = pd.read_csv(uploaded_file)
        
        # For display in HTML
        df_head_html = df.head().to_html(classes='table table-striped', index=False)

        percentage_change_text = None
        if 'Sales' in df.columns and 'Quarter' in df.columns:
            try:
                df['Sales'] = df['Sales'].astype(float)
                df['Quarter'] = pd.to_datetime(df['Quarter'])
                df = df.sort_values(by='Quarter')
                if len(df) >= 2:
                    last_two_quarters = df['Sales'].iloc[-2:]
                    percentage_change = ((last_two_quarters.iloc[1] - last_two_quarters.iloc[0]) / last_two_quarters.iloc[0]) * 100
                    percentage_change_text = f"نسبة التغير في المبيعات بين الربعين الأخيرين   {percentage_change:.2f}%"
                else:
                    percentage_change_text = "لا توجد بيانات كافية (أقل من ربعين) لحساب نسبة التغير."
            except Exception as e:
                percentage_change_text = f"تعذر حساب نسبة التغير في المبيعات: {e}"

        # LangChain Analysis
        summary_text = df.describe().to_string() 
        template = "حلل البيانات التالية وقدم ملخصًا احترافيًا: {text}"
        prompt = PromptTemplate.from_template(template)
        llm = Ollama(model="llama3")
        chain = LLMChain(prompt=prompt, llm=llm)
        llm_result = chain.invoke({"text": summary_text})

        return render_template('results.html', df_head=df_head_html, change_text=percentage_change_text, llm_result=llm_result['text'])

    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
