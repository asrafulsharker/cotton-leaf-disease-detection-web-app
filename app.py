from flask import Flask, render_template, request, url_for
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

import numpy as np
import os  


app = Flask(__name__)


dic = {
    0: {
        'name': 'Bacterial Blight',
        'description': 'Bacterial Blight is caused by bacteria that can infect the cotton plant, leading to water-soaked lesions on leaves and stems.',
        'suggestion': 'Farmers should remove infected plant parts, apply copper-based fungicides like *Bordeaux mixture*, and avoid overhead irrigation to prevent spreading. A recommended pesticide is *Streptomycin* for bacterial control.'
    },
    1: {
        'name': 'Curl Virus',
        'description': 'Curl Virus is a viral infection that causes leaves to curl and yellow, reducing the plant\'s ability to photosynthesize effectively.',
        'suggestion': 'Farmers should remove and destroy infected plants, control aphid populations with insecticides like *Imidacloprid* or *Acetamiprid*, and consider using resistant varieties if available.'
    },
    2: {
        'name': 'Healthy Leaf',
        'description': 'Healthy leaves are bright green and free from any visible disease symptoms, indicating the plant is growing well.',
        'suggestion': 'Farmers should continue with regular crop maintenance practices, such as watering, fertilizing, and monitoring for pests. No specific treatment is necessary for healthy leaves.'
    },
    3: {
        'name': 'Herbicide Growth Damage',
        'description': 'Herbicide damage occurs when herbicides are sprayed incorrectly, leading to stunted growth or leaf deformities.',
        'suggestion': 'Farmers should avoid applying herbicides during windy conditions and follow proper instructions to prevent damage to non-target plants. There are no specific medicines for herbicide damage; focus on proper herbicide application techniques.'
    },
    4: {
        'name': 'Leaf Hopper Jassids',
        'description': 'Leaf hopper Jassids are insects that feed on plant sap, leading to yellowing and stunting of plant growth.',
        'suggestion': 'Farmers should apply insecticides like *Dimethoate* or *Thiamethoxam* to control leaf hopper populations. Beneficial insects such as spiders and ladybugs can also help manage pest populations naturally.'
    },
    5: {
        'name': 'Leaf Redding',
        'description': 'Leaf redding is caused by various factors like nutrient deficiency or certain diseases, resulting in red or purple discoloration of the leaves.',
        'suggestion': 'Farmers should ensure proper nutrient management with balanced fertilizers. If caused by disease, apply fungicides such as *Mancozeb* or *Chlorothalonil*. Soil testing can help determine any deficiencies.'
    },
    6: {
        'name': 'Leaf Variegation',
        'description': 'Leaf variegation occurs when the leaves show irregular patterns of light and dark green, often due to viral infections or environmental stress.',
        'suggestion': 'Farmers should inspect plants for pests, diseases, and nutrient imbalances and remove any affected leaves. Use insecticides like *Imidacloprid* for controlling aphids and other pest vectors that spread the virus.'
    },
    7: {
        'name': 'Aphids',
        'description': 'Aphids are small insects that suck sap from plants, causing wilting, yellowing, and stunted growth.',
        'suggestion': 'Farmers can control aphid populations by introducing beneficial insects like ladybugs or by applying insecticidal soap or neem oil. Common insecticides include *Acetamiprid*, *Malathion*, and *Imidacloprid*.'
    },
    8: {
        'name': 'Army Worm',
        'description': 'Army worms are larvae of moths that can cause extensive damage to cotton plants by feeding on leaves and fruit.',
        'suggestion': 'Farmers should monitor for signs of army worms and apply appropriate insecticides like *Chlorpyrifos* or *Spinosad* if infestation levels are high. Early intervention is key.'
    },
    9: {
        'name': 'Fusarium Wilt',
        'description': 'Fusarium wilt is a fungal disease that causes yellowing and wilting of the leaves, often leading to plant death.',
        'suggestion': 'Farmers should rotate crops to break the disease cycle, remove infected plants, and apply fungicides such as *Fluopyram* or *Thiophanate-methyl* to manage the spread of the fungus.'
    },
    10: {
        'name': 'Target Spot',
        'description': 'Target spot is a fungal disease that causes dark spots with concentric rings on the leaves, which can lead to premature leaf drop.',
        'suggestion': 'Farmers should practice good crop rotation, remove infected leaves, and apply fungicides such as *Azoxystrobin* or *Tebuconazole* to reduce the impact of target spot.'
    },
    11: {
        'name': 'Powdery Mildew',
        'description': 'Powdery mildew is a fungal disease characterized by a white, powdery coating on the surface of leaves and stems.',
        'suggestion': 'Farmers should apply fungicides like *Sulfur* or *Myclobutanil*, increase airflow around the plants, and remove infected plant material to control the spread of powdery mildew.'
    }
}






model = load_model('cotton_disease_vgg16.h5')

model.make_predict_function()

def predict_label_with_description(img_path):
    i = load_img(img_path, target_size=(224, 224))  # Adjust to the size the model was trained with
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = preprocess_input(i)
    p = model.predict(i)
    predicted_class_index = np.argmax(p)
    predicted_class_info = dic[predicted_class_index]
    confidence = p[0][predicted_class_index] * 100
    suggestion = predicted_class_info['suggestion']  # Get suggestion based on the class
    return predicted_class_info, confidence, suggestion





@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Dragon FruitHarvest Master Your Ultimate Guide to Perfectly Timed Harvesting and Maturity Detection"

@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = os.path.join(app.root_path, "static", img.filename)
        img.save(img_path)
        predicted_class_info, confidence, suggestion = predict_label_with_description(img_path)
        return render_template("index.html", 
                               predicted_class_info=predicted_class_info, 
                               confidence=confidence, 
                               img_path=img.filename,
                               suggestion=suggestion)
    return render_template("index.html")


if __name__ =='__main__':
    app.run(debug=False, host='0.0.0.0')