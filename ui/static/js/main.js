document.addEventListener('DOMContentLoaded', () => {
    // ── Elements ─────────────────────────────────────────────────────────────
    const form = document.getElementById('diagnostic-form');
    const imageInput = document.getElementById('image');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImage = document.getElementById('image-preview');
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const analyzeBtn = document.getElementById('analyze-btn');
    const metadataToggle = document.getElementById('use-metadata');
    const metadataFields = document.getElementById('metadata-fields');

    const diagnosisText = document.getElementById('diagnosis-text');
    const confidenceRing = document.getElementById('confidence-ring');
    const confidenceText = document.getElementById('confidence-text');
    const heatmapImage = document.getElementById('heatmap-image');
    const reportContent = document.getElementById('llm-report');

    // ── Toggle Logic ─────────────────────────────────────────────────────────
    metadataToggle.addEventListener('change', () => {
        if (metadataToggle.checked) {
            metadataFields.classList.remove('disabled');
        } else {
            metadataFields.classList.add('disabled');
        }
    });

    // ── Drag & Drop Logic ────────────────────────────────────────────────────
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            showPreview();
        }
    });

    imageInput.addEventListener('change', showPreview);

    function showPreview() {
        if (imageInput.files && imageInput.files[0]) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                dropZone.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(imageInput.files[0]);
        }
    }

    // ── Form Submission to FastAPI ───────────────────────────────────────────
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!imageInput.files || !imageInput.files[0]) {
            alert("Please upload an MRI image first.");
            return;
        }

        // Prepare UI for loading
        analyzeBtn.disabled = true;
        analyzeBtn.innerText = "SCANNING...";
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Build FormData payload — only include metadata if toggle is ON
        const formData = new FormData();
        formData.append("image", imageInput.files[0]);

        if (metadataToggle.checked) {
            formData.append("age", document.getElementById('age').value);
            formData.append("mmse", document.getElementById('mmse').value);
            formData.append("cdr", document.getElementById('cdr').value);
            formData.append("education_years", document.getElementById('education_years').value);
            formData.append("apoe4", document.getElementById('apoe4').value);
        }
        // If toggle is OFF, fields are simply omitted — API will detect nulls and use MRI-only mode

        try {
            // Call the FastAPI backend
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Render Results
            renderResults(data);

        } catch (error) {
            console.error(error);
            alert("Failed to analyze. Ensure FastAPI backend is running on port 8000.");
        } finally {
            // Restore UI
            loadingState.classList.add('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "RUN DIAGNOSTIC SCAN";
        }
    });

    // ── Result Rendering ─────────────────────────────────────────────────────
    function renderResults(data) {
        // Show results section
        resultsSection.classList.remove('hidden');

        // Diagnosis
        diagnosisText.innerText = data.prediction;
        if (data.prediction.includes("NonDemented")) {
            diagnosisText.style.color = "var(--primary)";
        } else {
            diagnosisText.style.color = "var(--amber-warning)";
        }

        // Animate Radial Gauge (Confidence)
        // Math: circumference = 2 * pi * r = 2 * 3.14 * 40 = 251.2
        const percent = Math.round(data.confidence * 100);
        const offset = 251.2 - (251.2 * percent) / 100;
        
        // Reset animation
        confidenceRing.style.transition = "none";
        confidenceRing.style.strokeDashoffset = 251.2;
        
        // Trigger reflow
        confidenceRing.getBoundingClientRect(); 
        
        // Animate to new value
        confidenceRing.style.transition = "stroke-dashoffset 1s ease-out";
        confidenceRing.style.strokeDashoffset = offset;
        
        // Count up text
        let count = 0;
        const interval = setInterval(() => {
            if (count >= percent) {
                clearInterval(interval);
                confidenceText.innerText = `${percent}%`;
            } else {
                count++;
                confidenceText.innerText = `${count}%`;
            }
        }, 10);

        // Heatmap Image
        heatmapImage.src = `http://localhost:8000${data.heatmap_url}?t=${new Date().getTime()}`;

        // LLM Report
        reportContent.innerText = data.report;
    }
});
