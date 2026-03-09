<h1 align="center">🌿 AgriAI – AI Powered Plant Disease Detection</h1>

<p align="center">
AgriAI is a <b>full-stack AI system for plant disease detection</b> using <b>Deep Learning</b> and 
<b>LLM-based treatment recommendations</b>.
</p>

<p align="center">
The system combines <b>Computer Vision + Large Language Models + Web APIs</b> to help farmers detect plant diseases early and receive treatment guidance.
</p>

<hr>

<h2>🚀 Features</h2>

<ul>
<li>✔ Plant disease detection from <b>leaf images</b></li>
<li>✔ High accuracy CNN model (~98.9% validation accuracy)</li>
<li>✔ LLM-based treatment recommendation</li>
<li>✔ Organic and chemical treatment suggestions</li>
<li>✔ Prevention tips for farmers</li>
<li>✔ Real-time API prediction</li>
<li>✔ Web interface for uploading images</li>
</ul>

<hr>

<h2>🧠 Model (Level 2)</h2>

<p>
The <b>Level-2 model</b> improves generalization and real-world performance using advanced training strategies.
</p>

<h3>Architecture</h3>

<table border="1" cellpadding="8">
<tr>
<th>Component</th>
<th>Details</th>
</tr>

<tr>
<td>Model</td>
<td>MobileNetV2 (Transfer Learning)</td>
</tr>

<tr>
<td>Input Size</td>
<td>224 x 224</td>
</tr>

<tr>
<td>Framework</td>
<td>PyTorch</td>
</tr>

<tr>
<td>Task</td>
<td>Multi-class plant disease classification</td>
</tr>

</table>

<hr>

<h2>⚙️ Training Improvements</h2>

<ul>
<li>Stronger image augmentation</li>
<li>Cosine learning rate scheduler</li>
<li>Label smoothing</li>
<li>Dropout regularization</li>
<li>Weight decay</li>
<li>Balanced class weights</li>
<li>Fine-tuning deeper layers</li>
</ul>

<hr>

<h2>📊 Model Performance</h2>

<table border="1" cellpadding="8">
<tr>
<th>Metric</th>
<th>Score</th>
</tr>

<tr>
<td>Training Accuracy</td>
<td><b>98.63%</b></td>
</tr>

<tr>
<td>Validation Accuracy</td>
<td><b>98.97%</b></td>
</tr>

</table>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d65cbeaf-0353-4bf9-9e39-6c0cb7fd06d1" width="48%">
<img src="https://github.com/user-attachments/assets/8250311c-99eb-4254-a4c0-586e1be508fb" width="48%">
</p>
<hr>

<h2>🌱 Supported Crops</h2>

<p>
The model is trained only on the following crop types. Predictions are reliable only for these crops.
</p>

<ul>
<li>Apple</li>
<li>Blueberry</li>
<li>Cherry</li>
<li>Corn (Maize)</li>
<li>Potato</li>
<li>Rice</li>
<li>Tomato</li>
</ul>

<p>
⚠️ If an image from a crop outside this list is uploaded, the system may return 
<b>"Uncertain Prediction"</b> or produce inaccurate results.
</p>
<hr>

<h2>⚠️ When Results May Not Be Displayed</h2>

<p>
The system may return <b>“Uncertain Prediction”</b> or may not display results in certain conditions to avoid incorrect diagnosis.
</p>

<h3>1️⃣ Low Model Confidence</h3>

<p>
If the prediction confidence is below <b>65%</b>, the system returns:
</p>

<pre>
Uncertain Prediction
</pre>

<p>
This prevents misleading disease predictions.
</p>

<h3>2️⃣ Poor Image Quality</h3>

<ul>
<li>Blurry image</li>
<li>Low resolution</li>
<li>Too dark or too bright</li>
<li>Too much background</li>
</ul>

<p>
The model performs best when the <b>leaf is clearly visible</b>.
</p>

<h3>3️⃣ Unsupported Crop Type</h3>

<p>
The model is trained only on specific crops and diseases present in the dataset.  
If a different plant is uploaded, predictions may be uncertain.
</p>

<h3>4️⃣ Multiple Leaves / Complex Background</h3>

<ul>
<li>Multiple leaves</li>
<li>Soil or other objects</li>
<li>Partially visible leaves</li>
</ul>

<p>
These may reduce prediction accuracy.
</p>

<h3>5️⃣ External API Failure</h3>

<p>
Treatment recommendations are generated using the <b>Groq LLM API</b>.
</p>

<p>
If the API fails due to:
</p>

<ul>
<li>Network issues</li>
<li>API key problems</li>
<li>Rate limits</li>
</ul>

<p>
The system automatically returns a <b>fallback treatment response</b>.
</p>

<hr>

<h2>🛠 Tech Stack</h2>

<ul>
<li><b>AI / ML:</b> PyTorch, MobileNetV2, Computer Vision</li>
<li><b>Backend:</b> FastAPI, Python</li>
<li><b>Frontend:</b> HTML, CSS, JavaScript</li>
<li><b>AI Services:</b> Groq LLM API</li>
</ul>

<hr>

<h2>🎯 Future Improvements</h2>

<ul>
<li>Support for more crop types</li>
<li>Mobile app for farmers</li>
<li>Real-time disease detection using camera</li>
<li>Multilingual farmer support</li>
<li>Field deployment optimization</li>
</ul>
