(function () {
    'use strict';

    const $ = (s) => document.querySelector(s);
    const dropZone = $('#dropZone'), fileInput = $('#fileInput'), dropLabel = $('#dropLabel');
    const confSlider = $('#confidence'), confVal = $('#confVal'), btnRun = $('#btnRun');
    const frameSkipSlider = $('#frameSkip'), frameSkipVal = $('#frameSkipVal'), frameSkipLabel = $('#frameSkipLabel');
    const loading = $('#loading'), loadingText = $('#loadingText');
    const progress = $('#progress'), progressFill = $('#progressFill'), progressText = $('#progressText');
    const resultArea = $('#resultArea'), imageResult = $('#imageResult'), videoResult = $('#videoResult');
    const resultImg = $('#resultImg'), resultVideo = $('#resultVideo'), downloadLink = $('#downloadLink');
    const errorToast = $('#errorToast'), errorMessage = $('#errorMessage');

    let selectedFile = null;

    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => errorToast.classList.add('hidden'), 6000);
    }

    function isImage(file) { return file.type.startsWith('image/'); }
    function isVideo(file) { return file.type.startsWith('video/'); }

    confSlider.addEventListener('input', () => { confVal.textContent = confSlider.value + '%'; });
    frameSkipSlider.addEventListener('input', () => { frameSkipVal.textContent = frameSkipSlider.value; });

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]); });
    fileInput.addEventListener('change', () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); });

    function selectFile(file) {
        if (!isImage(file) && !isVideo(file)) { showError('Please select an image or video file.'); return; }
        selectedFile = file;
        dropZone.classList.add('has-file');
        dropLabel.textContent = file.name;
        btnRun.disabled = false;
        resultArea.classList.add('hidden');
        if (isVideo(file)) frameSkipLabel.classList.remove('hidden');
        else frameSkipLabel.classList.add('hidden');
    }

    btnRun.addEventListener('click', () => {
        if (!selectedFile) return;
        if (isImage(selectedFile)) runImageDetection();
        else runVideoInspection();
    });

    function runImageDetection() {
        resultArea.classList.add('hidden');
        loading.classList.remove('hidden');
        loadingText.textContent = 'Analyzing image... (may take 30-60s on first run)';
        btnRun.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        const xhr = new XMLHttpRequest();
        xhr.responseType = 'blob';

        xhr.addEventListener('load', () => {
            loading.classList.add('hidden');
            btnRun.disabled = false;
            if (xhr.status >= 200 && xhr.status < 300) {
                const blob = xhr.response;
                const url = URL.createObjectURL(blob);
                resultImg.src = url;
                downloadLink.href = url;
                downloadLink.download = 'flashinspector_result.jpg';
                imageResult.classList.remove('hidden');
                videoResult.classList.add('hidden');
                resultArea.classList.remove('hidden');
            } else {
                showError('Detection failed: HTTP ' + xhr.status);
            }
        });

        xhr.addEventListener('error', () => {
            loading.classList.add('hidden');
            btnRun.disabled = false;
            showError('Network error. The server may still be starting up — try again in a minute.');
        });

        xhr.addEventListener('timeout', () => {
            loading.classList.add('hidden');
            btnRun.disabled = false;
            showError('Request timed out. The server may need more time to load the model — try again.');
        });

        xhr.open('POST', '/detect?confidence=' + confSlider.value);
        xhr.timeout = 120000;
        xhr.send(formData);
    }

    function runVideoInspection() {
        resultArea.classList.add('hidden');
        progress.classList.remove('hidden');
        loading.classList.add('hidden');
        btnRun.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        const url = '/inspect/video?confidence=' + confSlider.value + '&frame_skip=' + frameSkipSlider.value;

        const xhr = new XMLHttpRequest();
        xhr.responseType = 'blob';

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = pct + '%';
                progressText.textContent = 'Uploading... ' + pct + '%';
            }
        });

        xhr.upload.addEventListener('load', () => {
            progressFill.style.width = '100%';
            progressText.textContent = 'Upload done. Processing video...';
            loading.classList.remove('hidden');
            loadingText.textContent = 'Running detection on every frame... This may take a while.';
        });

        xhr.addEventListener('load', () => {
            loading.classList.add('hidden');
            progress.classList.add('hidden');
            btnRun.disabled = false;
            if (xhr.status >= 200 && xhr.status < 300) {
                const blob = xhr.response;
                const blobUrl = URL.createObjectURL(blob);
                resultVideo.src = blobUrl;
                downloadLink.href = blobUrl;
                downloadLink.download = 'flashinspector_result.mp4';
                videoResult.classList.remove('hidden');
                imageResult.classList.add('hidden');
                resultArea.classList.remove('hidden');
            } else {
                showError('Video inspection failed: HTTP ' + xhr.status);
            }
        });

        xhr.addEventListener('error', () => {
            loading.classList.add('hidden');
            progress.classList.add('hidden');
            btnRun.disabled = false;
            showError('Network error. The server may still be starting up — try again in a minute.');
        });

        xhr.addEventListener('timeout', () => {
            loading.classList.add('hidden');
            progress.classList.add('hidden');
            btnRun.disabled = false;
            showError('Request timed out. Try a shorter video or increase frame skip.');
        });

        xhr.open('POST', url);
        xhr.timeout = 600000;
        xhr.send(formData);
    }
})();
