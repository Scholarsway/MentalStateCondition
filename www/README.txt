MENTAL HEALTH MLP PREDICTION APP
================================

This is a browser/mobile-ready prototype using:
- HTML
- CSS
- JavaScript
- TensorFlow.js
- The trained 7-class MLP model

FILES
-----
index.html
style.css
app.js
model/model.json
model/group1-shard1of1.bin
model/preprocessing.json
model/class_mapping.json

IMPORTANT
---------
Do NOT open index.html by double-clicking it. Browser security may block local model files.
Run a local web server in this folder.

VS CODE METHOD
--------------
1. Open this folder in VS Code.
2. Install the "Live Server" extension if needed.
3. Right-click index.html.
4. Select "Open with Live Server".

OR WITH PYTHON
--------------
Open a terminal in this folder and run:

python -m http.server 8000

Then open:
http://localhost:8000

MODEL INPUT
-----------
The application collects 10 clinical features. After preprocessing they become 13 numerical inputs to the MLP because the three categorical variables are one-hot encoded.

OUTPUT CLASSES
--------------
0 Healthy
1 Depression
2 Anxiety Disorder
3 Bipolar Disorder
4 Schizophrenia
5 Post-Traumatic Stress Disorder (PTSD)
6 Obsessive-Compulsive Disorder (OCD)

NOTE
----
The current prototype loads TensorFlow.js from a CDN. Therefore the first browser test requires internet access. When we package the app for offline Android use, TensorFlow.js itself will be bundled locally.
