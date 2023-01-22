from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import csv

app = Flask(__name__)

model = load_model('models\Modelnc2.h5')

model.make_predict_function()


def data_read(x):
    with open('static\datasheet\Datasheet.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for data in reader:
            if data[0] == str(x):
                return data[1:]


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100, 100))
    i = image.img_to_array(i)/255
    i = i.reshape(1, 100, 100, 3)
    p = model.predict_classes(i)
    return data_read(p[0])


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "PBL Sem 4"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['image']

        img_path = "uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
