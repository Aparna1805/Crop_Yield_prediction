from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'crop_yield_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def language_select():
    return render_template('language.html')

@app.route('/predictor', methods=['POST'])
def predictor():
    lang = request.form.get('language')
    return redirect(url_for('predict', lang=lang))

@app.route('/predict/<lang>', methods=['GET', 'POST'])
def predict(lang):
    if request.method == 'POST':
        try:
            rainfall = float(request.form['rainfall'])
            soil_quality = float(request.form['soil_quality'])
            farm_size = float(request.form['farm_size'])
            sunlight = float(request.form['sunlight'])
            fertilizer = float(request.form['fertilizer'])

            input_data = np.array([[rainfall, soil_quality, farm_size, sunlight, fertilizer]])
            prediction = model.predict(input_data)[0]

            if lang == 'ta':
                return render_template('index_ta.html', prediction_text=f'முன்னறியப்பட்ட பயிர் மகசூல்: {prediction:.2f}', lang=lang)
            return render_template('index_en.html', prediction_text=f'Predicted Crop Yield: {prediction:.2f}', lang=lang)
        except:
            msg = "தவறான தகவல்!" if lang == 'ta' else "⚠️ Please enter valid inputs."
            template = 'index_ta.html' if lang == 'ta' else 'index_en.html'
            return render_template(template, prediction_text=msg, lang=lang)

    return render_template('index_ta.html' if lang == 'ta' else 'index_en.html', lang=lang)

@app.route('/thankyou/<lang>')
def thankyou(lang):
    template = 'thankyou.html' if lang == 'en' else 'thankyou_ta.html'
    return render_template(template)

if __name__ == '__main__':
    app.run(debug=True)
