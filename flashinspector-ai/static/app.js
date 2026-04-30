(function () {
    'use strict';

    const $ = (s) => document.querySelector(s);
    const dropZone = $('#dropZone'), fileInput = $('#fileInput'), dropLabel = $('#dropLabel');
    const confSlider = $('#confidence'), confVal = $('#confVal'), btnRun = $('#btnRun');
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
    }

    btnRun.addEventListener('click', () => {
        if (!selectedFile) return;
        if (isImage(selectedFile)) runImageDetection();
        else runVideoInspection();
    });

    async function runImageDetection() {
        resultArea.classList.add('hidden');
        loading.classList.remove('hidden');
        loadingText.textContent = 'Running detection...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('/detect?confidence=' + confSlider.value, { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Server returned ' + res.status);
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);

            resultImg.src = url;
            downloadLink.href = url;
            downloadLink.download = 'flashinspector_result.jpg';
            imageResult.classList.remove('hidden');
            videoResult.classList.add('hidden');
            resultArea.classList.remove('hidden');
        } catch (e) {
            showError('Detection failed: ' + e.message);
        } finally {
            loading.classList.add('hidden');
        }
    }

    function runVideoInspection() {
        resultArea.classList.add('hidden');
        progress.classList.remove('hidden');
        loading.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);
        const url = '/inspect/video?confidence=' + confSlider.value + '&frame_skip=3';

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
            showError('Network error.');
        });

        xhr.addEventListener('timeout', () => {
            loading.classList.add('hidden');
            progress.classList.add('hidden');
            showError('Request timed out. Try a shorter video.');
        });

        xhr.open('POST', url);
        xhr.timeout = 600000;
        xhr.send(formData);
    }
})();
