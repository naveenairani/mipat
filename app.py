import os
import secrets
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import grayscale_to_color
from yolo import load_yolo_model, detect_objects
from displayTumor import DisplayTumor
from predictTumor import load_finetune_model, predict_tumor

app = Flask(__name__)
secret_key = secrets.token_hex(32)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Load models
net, labels, colors = load_yolo_model()
yolo_model = YOLO("runs/detect/train/weights/best.pt")
brain_tumor_model = load_finetune_model()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for grayscale to color
@app.route('/grayscale_to_color', methods=['GET', 'POST'])
def grayscale_to_color_route():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                colored_img = grayscale_to_color(img)
                original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
                colored_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'colored_image.jpg')
                cv2.imwrite(original_img_path, img)
                cv2.imwrite(colored_img_path, colored_img)
                return render_template('grayscale_to_color.html', success=True, image_path=colored_img_path, original_image_path=original_img_path)
            else:
                return render_template('grayscale_to_color.html', error='Error reading the image')
        else:
            return render_template('grayscale_to_color.html', error='No image uploaded')
    return render_template('grayscale_to_color.html')

# Route for YOLO
@app.route('/yolo')
def yolo():
    return render_template('yolo.html')

# Route to upload file for YOLO
@app.route('/yolo_upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        output_path = detect_objects(filepath, net, labels, colors)
        return redirect(url_for('yolo_result', filename='output.jpg'))

# Route to display YOLO result
@app.route('/yolo_result')
def yolo_result():
    filename = request.args.get('filename')
    return render_template('yolo_result.html', filename=filename)

# Route for new page 3
@app.route('/newpage3')
def newpage3():
    return render_template('newpage3.html')

# Route for new page 4
@app.route('/newpage4')
def newpage4():
    return render_template('newpage4.html')

# Route for brain tumor detection
@app.route('/newpage5')
def newpage5():
    return render_template('index_brain.html')

# Upload route for brain tumor detection
@app.route('/upload', methods=['POST'])
def upload_brain_tumor_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        image = Image.open(file.stream).convert("RGB")
        img_array = np.array(image)
        dt = DisplayTumor()
        dt.readImage(image)
        dt.removeNoise()
        dt.displayTumor()
        processed_img = dt.getImage()
        prediction = predict_tumor(brain_tumor_model, img_array)
        processed_img_pil = Image.fromarray(processed_img)
        processed_img_pil.save("static/processed_image.jpg")
        return render_template('result_brain.html', prediction=prediction[0][0], image_url=url_for('static', filename='processed_image.jpg'))

# Route for kidney stone detection page
@app.route('/newpage6', methods=['GET', 'POST'])
def newpage6():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = yolo_model.predict(source=image)
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        label = yolo_model.names[class_id] if class_id < len(yolo_model.names) else "Unknown"
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + file.filename)
            cv2.imwrite(detected_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return render_template('kidney.html', image_url=url_for('static', filename='uploads/' + file.filename), detected_image_url=url_for('static', filename='uploads/detected_' + file.filename))
    return render_template('kidney.html')

# Route for new page 7
@app.route('/newpage7')
def newpage7():
    return render_template('newpage7.html')

# Route for new page 8
@app.route('/newpage8')
def newpage8():
    return render_template('newpage8.html')

if __name__ == '__main__':
    app.run(debug=True)
