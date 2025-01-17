# Import necessary libraries
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobileV2
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)

app = Flask(__name__)

dic = {
    0: {
        'name': 'Aphids',
        'description': 'Aphids are small insects that can infest cotton plants. They typically occur due to favorable warm conditions and reproduce rapidly. To manage aphid infestations, consider early detection, natural predators like ladybugs and parasitic wasps, beneficial insects, insecticidal soaps or oils, and chemical pesticides as a last resort. Additionally, practices like crop rotation, weed control, and monitoring weather conditions can help prevent aphid-related issues in cotton farming.'
    },
    1: {
        'name': 'Armyworms',
        'description': 'Armyworms are destructive pests affecting cotton crops. These caterpillars can rapidly devour leaves and damage cotton plants. They are known for their voracious feeding habits and can lead to significant yield losses. To manage armyworm infestations, use integrated pest management strategies, including natural enemies, biological control agents, and selective insecticides when necessary. Monitoring and early intervention are key to preventing extensive damage to cotton fields.'
    },
    2: {
        'name': 'Bacterial Blight',
        'description': 'Bacterial blight is a cotton disease caused by bacterial pathogens. It manifests as dark lesions on leaves and can lead to defoliation, reducing cotton yield and quality. Managing bacterial blight involves planting disease-resistant cotton varieties, practicing crop rotation, and using copper-based fungicides when necessary. Proper field sanitation and avoiding excessive moisture can also help prevent the disease.'
    },
    3: {
        'name': 'Cotton Boll Rot',
        'description': 'Cotton boll rot is a fungal disease that damages cotton bolls, leading to premature cotton fiber release and reduced yield. It is managed by using disease-resistant cotton varieties and maintaining proper plant spacing to reduce humidity. Fungicides can be applied if needed, but prevention is key.'
    },
    4: {
        'name': 'Curl Virus',
        'description': 'Curl virus, also known as Cotton Leaf Curl Disease, is a viral infection that affects cotton plants. It is transmitted by whiteflies and causes the leaves of cotton plants to curl and become deformed. This disease can significantly reduce cotton yields. Control measures include using virus-resistant cotton varieties, managing whitefly populations, and practicing crop rotation.'
    },
    5: {
        'name': 'Fusarium Wilt',
        'description': 'Fusarium wilt is a plant disease caused by the Fusarium fungus. It affects various crops, including cotton. The fungus infects the plant\'s vascular system, leading to symptoms such as wilting, yellowing of leaves, and stunted growth. Fusarium wilt can result in reduced crop yields and quality. Prevention methods include using disease-resistant cotton varieties, practicing crop rotation, and ensuring proper soil health and drainage.'
    },
    6: {
        'name': 'Green Cotton Boll',
        'description': 'Green cotton boll refers to the stage in the cotton plant\'s growth cycle when cotton bolls, which contain cotton fibers, are still immature and green in color. At this stage, the cotton fibers inside the boll are not fully developed or ready for harvesting. It is crucial for cotton growers to monitor the maturation of the cotton bolls and pick them at the right time to obtain high-quality cotton fibers for textile production. Harvesting green cotton bolls can result in lower fiber quality and reduced yields. Proper timing of cotton boll harvesting is essential for the success of cotton farming.'
    },
    7: {
        'name': 'Healthy',
        'description': 'The term "healthy" in the context of cotton farming refers to cotton plants that are free from diseases, pests, and other stressors. Healthy cotton plants exhibit vigorous growth, vibrant green foliage, and well-developed cotton bolls. They are able to reach their full yield potential, producing high-quality cotton fibers that are suitable for various industrial and textile applications. Maintaining the health of cotton crops often involves regular monitoring, proper irrigation, timely fertilization, and effective pest and disease management practices to ensure a successful and productive cotton harvest.'
    },
    8: {
        'name': 'Powdery Mildew',
        'description': 'Powdery mildew is a fungal disease affecting cotton. It appears as white powdery spots on leaves and can reduce photosynthesis and cotton quality. Management includes fungicides, proper spacing, resistant varieties, crop rotation, monitoring, and disease forecasting.'
    },
    9: {
        'name': 'Target Spot',
        'description': 'Target spot is a fungal disease in cotton. It causes circular lesions on leaves with a distinctive target-like appearance. It can reduce cotton yield and quality. Control measures include fungicides, crop rotation, and planting resistant varieties. Proper irrigation and field hygiene are also essential for management.'
    }
}


model = load_model('merged_model.h5')  # Load your merged model here

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_resnet = image.img_to_array(img)
    img_resnet = np.expand_dims(img_resnet, axis=0)
    img_resnet = preprocess_input_resnet(img_resnet)

    img_mobileV2 = image.img_to_array(img)
    img_mobileV2 = np.expand_dims(img_mobileV2, axis=0)
    img_mobileV2 = preprocess_input_mobileV2(img_mobileV2)

    # Assuming the model takes two inputs
    return [img_resnet, img_mobileV2]

def predict_label_with_description(img_path):
    images = preprocess_image(img_path)
    p = model.predict(images)  # This should work if model is designed for multiple inputs
    predicted_class_index = np.argmax(p)
    predicted_class_info = dic[predicted_class_index]
    confidence = p[0][predicted_class_index] * 100
    return predicted_class_info, confidence


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Cotton FruitHarvest Master Your Ultimate Guide to Perfectly Timed Harvesting and Maturity Detection"

@app.route("/submit", methods=["POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = os.path.join(app.root_path, "static", img.filename)
        img.save(img_path)
        predicted_class_info, confidence = predict_label_with_description(img_path)
        return render_template("index.html", predicted_class_info=predicted_class_info, confidence=confidence, img_path=img.filename)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
