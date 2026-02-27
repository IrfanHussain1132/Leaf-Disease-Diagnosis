// ==============================
// Initialize Lucide Icons
// ==============================
lucide.createIcons();

// ==============================
// DOM Elements
// ==============================
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('results');

let selectedFile = null;

// ==============================
// 1️⃣ Image Upload Logic
// ==============================

dropZone.addEventListener('click', (e) => {
    if (e.target === dropZone) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', function () {
    handleFiles(this.files);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('bg-green-100');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('bg-green-100');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('bg-green-100');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPlaceholder.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(selectedFile);
    }
}

// ==============================
// 2️⃣ Reset Logic
// ==============================

resetBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    previewContainer.classList.add('hidden');
    uploadPlaceholder.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    fileInput.value = '';
    selectedFile = null;
});

// ==============================
// 3️⃣ AI ANALYSIS (Connected to FastAPI)
// ==============================

analyzeBtn.addEventListener('click', async () => {

    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    analyzeBtn.classList.add('hidden');
    resetBtn.classList.add('hidden');
    loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        resetBtn.classList.remove('hidden');

        displayResults(data);

    } catch (error) {
        loader.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        resetBtn.classList.remove('hidden');

        alert("Error connecting to backend.");
        console.error(error);
    }
});

// ==============================
// 4️⃣ Display Results (REAL + SAFE)
// ==============================

function displayResults(data) {

    resultsSection.classList.remove('hidden');

    // Disease & Confidence
    // Clean disease name (remove ___ and _)
    const cleanName = (data.disease || "Unknown")
        .replace(/___/g, " ")
        .replace(/_/g, " ");

    document.getElementById('res-disease').innerText = cleanName;

    document.getElementById('res-confidence-text').innerText =
        (data.confidence || 0) + "%";

    document.getElementById('res-confidence-bar').style.width =
        (data.confidence || 0) + "%";

    // Traditional Treatment
    document.getElementById('res-trad-method').innerText =
        data.traditional?.method || "Not available";

    document.getElementById('res-trad-freq').innerText =
        data.traditional?.frequency || "Not available";

    document.getElementById('res-trad-effect').innerText =
        data.traditional?.effect || "Not available";

    // Chemical Treatment
    document.getElementById('res-chem-name').innerText =
        data.chemical?.name || "Not available";

    document.getElementById('res-chem-dosage').innerText =
        data.chemical?.dosage || "Not available";

    document.getElementById('res-chem-interval').innerText =
        data.chemical?.interval || "Not available";

    document.getElementById('res-chem-timeline').innerText =
        data.chemical?.timeline || "Not available";

    // Prevention List
    const prevList = document.getElementById('res-prevention');
    prevList.innerHTML = "";

    const tips = data.prevention || [];

    tips.forEach(tip => {
        const li = document.createElement('li');
        li.className = "flex items-start gap-3 text-slate-700 text-sm";
        li.innerHTML = `
            <div class="mt-1.5 h-2 w-2 rounded-full bg-green-500 shrink-0"></div>
            <span>${tip}</span>
        `;
        prevList.appendChild(li);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}