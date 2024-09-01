import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime

# Load the pre-trained model
model = load_model('final_roller_coaster_model.keras')

# Function to predict and process the image
def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (48, 48))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        gender_pred, age_pred = model.predict(image_input)

        gender = 'Female' if gender_pred[0][0] < 0.5 else 'Male'
        age = int(age_pred[0][0])

        if age < 13 or age > 60:
            cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 10)
            result_text.set(f"Not Allowed\nAge: {age}, Gender: {gender}")
        else:
            result_text.set(f"Allowed\nAge: {age}, Gender: {gender}")

        # Save data to CSV
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {'Age': [age], 'Gender': [gender], 'Entry Time': [entry_time]}
        df = pd.DataFrame(data)
        df.to_csv('roller_coaster_data.csv', mode='a', header=False, index=False)

        # Update the image in the GUI
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# GUI setup
root = tk.Tk()
root.title("Horror Roller Coaster Age Detection")
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack()

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Upload Image and Predict", command=load_and_predict_image)
btn.pack()

root.mainloop()
