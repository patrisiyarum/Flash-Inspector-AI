/* FlashInspector AI — Client Application */
(function () {
    'use strict';

    const VIOLATION_CLASSES = new Set([
        'empty_mount', 'extinguisher_cabinet_empty', 'bracket_empty',
        'non_compliant_tag', 'noncompliant_tag', 'yellow_tag', 'red_tag',
        'exit_sign_dark', 'exit_dark', 'unlit_exit',
        'smoke_detector_missing', 'detector_missing',
        'blocked_exit', 'exit_blocked',
    ]);

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const apiUrlInput = $('#apiUrl'), btnHealth = $('#btnHealthCheck'), healthStatus = $('#healthStatus');
    const imageDropZone = $('#imageDropZone'), imageInput = $('#imageInput');
    const imageConfSlider = $('#imageConfidence'), imageConfVal = $('#imageConfVal');
    const btnDetect = $('#btnDetectImage'), imageResults = $('#imageResults');
    const imageLoading = $('#imageLoading'), canvas = $('#detectionCanvas');
    const equipmentList = $('#equipmentList'), violationList = $('#violationList');
    const imageInfTime = $('#imageInferenceTime');
    const videoDropZone = $('#videoDropZone'), videoInput = $('#videoInput');
    const videoConfSlider = $('#videoConfidence'), videoConfVal = $('#videoConfVal');
    const videoFrameSkip = $('#videoFrameSkip'), btnInspect = $('#btnInspectVideo');
    const videoProgress = $('#videoProgress'), videoProgressFill = $('#videoProgressFill');
    const videoProgressText = $('#videoProgressText'), videoResults = $('#videoResults');
    const videoLoading = $('#videoLoading'), videoInfTime = $('#videoInferenceTime');
    const errorToast = $('#errorToast'), errorMessage = $('#errorMessage');

    let selectedImage = null, selectedVideo = null;

    function getApiUrl() { return apiUrlInput.value.trim() || window.location.origin; }
    apiUrlInput.placeholder = window.location.origin;

    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => errorToast.classList.add('hidden'), 8000);
    }

    function isViolationClass(cls) { return VIOLATION_CLASSES.has(cls); }

    function formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = (sec % 60).toFixed(1);
        return m > 0 ? m + 'm ' + s + 's' : s + 's';
    }

    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('active'));
            $$('.tab-content').forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            $('#tab-' + tab.getAttribute('data-tab')).classList.add('active');
        });
    });

    imageConfSlider.addEventListener('input', () => { imageConfVal.textContent = imageConfSlider.value + '%'; });
    videoConfSlider.addEventListener('input', () => { videoConfVal.textContent = videoConfSlider.value + '%'; });

    btnHealth.addEventListener('click', async () => {
        healthStatus.textContent = '...';
        healthStatus.className = 'health-status';
        try {
            const res = await fetch(getApiUrl() + '/health', { signal: AbortSignal.timeout(5000) });
            if (res.ok) { healthStatus.textContent = 'Connected'; healthStatus.className = 'health-status health-ok'; }
            else { healthStatus.textContent = 'Error ' + res.status; healthStatus.className = 'health-status health-err'; }
        } catch (e) { healthStatus.textContent = 'Unreachable'; healthStatus.className = 'health-status health-err'; }
    });

    function setupDropZone(zone, input, onFile) {
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
        zone.addEventListener('dragleave', () => { zone.classList.remove('drag-over'); });
        zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
        input.addEventListener('change', () => { if (input.files[0]) onFile(input.files[0]); });
    }

    function onImageSelected(file) {
        if (!file.type.startsWith('image/')) { showError('Please select an image file.'); return; }
        selectedImage = file;
        imageDropZone.classList.add('has-file');
        imageDropZone.querySelector('p').textContent = file.name;
        btnDetect.disabled = false;
    }

    function onVideoSelected(file) {
        if (!file.type.startsWith('video/')) { showError('Please select a video file.'); return; }
        selectedVideo = file;
        videoDropZone.classList.add('has-file');
        videoDropZone.querySelector('p').textContent = file.name;
        btnInspect.disabled = false;
    }

    setupDropZone(imageDropZone, imageInput, onImageSelected);
    setupDropZone(videoDropZone, videoInput, onVideoSelected);

    btnDetect.addEventListener('click', async () => {
        if (!selectedImage) return;
        imageResults.classList.add('hidden');
        imageLoading.classList.remove('hidden');
        const formData = new FormData();
        formData.append('file', selectedImage);
        try {
            const res = await fetch(getApiUrl() + '/detect?confidence=' + imageConfSlider.value, { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Server returned ' + res.status);
            renderImageResults(await res.json());
        } catch (e) { showError('Detection failed: ' + e.message); }
        finally { imageLoading.classList.add('hidden'); }
    });

    function renderImageResults(data) {
        imageInfTime.textContent = 'Inference: ' + data.inference_time_ms + ' ms';
        drawDetections(data);
        const equipment = data.predictions.filter(p => !isViolationClass(p.class));
        const violations = data.violations || [];
        equipmentList.innerHTML = equipment.length === 0 ? '<div class="no-results">No equipment detected</div>' :
            equipment.map(p => '<div class="detection-item detection-item-equip"><span class="detection-class">' + p.class + '</span><span class="detection-conf">' + (p.confidence * 100).toFixed(1) + '%</span></div>').join('');
        violationList.innerHTML = violations.length === 0 ? '<div class="no-results">No violations found</div>' :
            violations.map(v => {
                const sc = v.severity === 'critical' ? 'detection-item-violation' : 'detection-item-warning';
                return '<div class="detection-item ' + sc + '"><div><span class="violation-label">' + v.label + '</span> <span class="severity-badge severity-' + v.severity + '">' + v.severity + '</span></div><span class="detection-conf">' + (v.confidence * 100).toFixed(1) + '%</span></div>';
            }).join('');
        imageResults.classList.remove('hidden');
    }

    function drawDetections(data) {
        const img = new Image();
        const reader = new FileReader();
        reader.onload = function (e) {
            img.onload = function () {
                const maxW = Math.min(1060, window.innerWidth - 60);
                const scale = Math.min(1, maxW / data.image.width);
                const dispW = Math.round(data.image.width * scale);
                const dispH = Math.round(data.image.height * scale);
                canvas.width = dispW; canvas.height = dispH;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, dispW, dispH);
                data.predictions.forEach(pred => {
                    const isViol = isViolationClass(pred.class);
                    const color = isViol ? '#ef4444' : '#4ade80';
                    const bgColor = isViol ? 'rgba(239,68,68,0.15)' : 'rgba(74,222,128,0.12)';
                    const x1 = pred.bbox.x1 * scale, y1 = pred.bbox.y1 * scale;
                    const x2 = pred.bbox.x2 * scale, y2 = pred.bbox.y2 * scale;
                    ctx.fillStyle = bgColor; ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    const label = pred.class + ' ' + (pred.confidence * 100).toFixed(0) + '%';
                    ctx.font = 'bold ' + Math.max(11, Math.round(12 * scale)) + 'px sans-serif';
                    const tw = ctx.measureText(label).width;
                    const lh = Math.max(16, Math.round(18 * scale));
                    const ly = Math.max(y1 - lh, 0);
                    ctx.fillStyle = color; ctx.fillRect(x1, ly, tw + 8, lh);
                    ctx.fillStyle = '#000'; ctx.textBaseline = 'top'; ctx.fillText(label, x1 + 4, ly + 2);
                });
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(selectedImage);
    }

    btnInspect.addEventListener('click', async () => {
        if (!selectedVideo) return;
        videoResults.classList.add('hidden');
        videoProgress.classList.remove('hidden');
        videoLoading.classList.add('hidden');
        const formData = new FormData();
        formData.append('file', selectedVideo);
        const url = getApiUrl() + '/inspect/video?confidence=' + videoConfSlider.value + '&frame_skip=' + videoFrameSkip.value;
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                videoProgressFill.style.width = pct + '%';
                videoProgressText.textContent = 'Uploading... ' + pct + '%';
            }
        });
        xhr.upload.addEventListener('load', () => {
            videoProgressFill.style.width = '100%';
            videoProgressText.textContent = 'Upload complete. Running inspection...';
            videoLoading.classList.remove('hidden');
        });
        xhr.addEventListener('load', () => {
            videoLoading.classList.add('hidden'); videoProgress.classList.add('hidden');
            if (xhr.status >= 200 && xhr.status < 300) {
                try { renderVideoResults(JSON.parse(xhr.responseText)); }
                catch (e) { showError('Failed to parse response.'); }
            } else { showError('Video inspection failed: HTTP ' + xhr.status); }
        });
        xhr.addEventListener('error', () => { videoLoading.classList.add('hidden'); videoProgress.classList.add('hidden'); showError('Network error.'); });
        xhr.addEventListener('timeout', () => { videoLoading.classList.add('hidden'); videoProgress.classList.add('hidden'); showError('Timed out. Try a shorter video.'); });
        xhr.open('POST', url); xhr.timeout = 600000; xhr.send(formData);
    });

    function renderVideoResults(data) {
        videoInfTime.textContent = 'Total inference: ' + data.inference_time_sec + 's';
        $('#vDuration').textContent = data.duration_sec;
        $('#vFrames').textContent = data.frames_processed;
        $('#vEquipment').textContent = data.equipment_count;
        $('#vViolations').textContent = data.total_violations;
        const tracks = data.tracks || [];
        const equipTracks = tracks.filter(t => !isViolationClass(t.class));
        const etb = $('#equipmentTable tbody');
        etb.innerHTML = equipTracks.length === 0 ? '<tr><td colspan="6" class="no-results">No equipment tracked</td></tr>' :
            equipTracks.map(t => '<tr><td>' + t.track_id + '</td><td><strong>' + t.class + '</strong></td><td>' + (t.best_confidence * 100).toFixed(1) + '%</td><td>' + formatTime(t.first_seen_sec) + '</td><td>' + formatTime(t.last_seen_sec) + '</td><td>' + t.hit_count + '</td></tr>').join('');
        const violations = data.violations || [];
        const vtb = $('#violationTable tbody');
        if (violations.length === 0) {
            vtb.innerHTML = '<tr><td colspan="5" class="no-results">No violations detected</td></tr>';
        } else {
            const seen = new Set();
            const uv = violations.filter(v => { const k = v.type + '|' + v.timestamp_sec; if (seen.has(k)) return false; seen.add(k); return true; });
            vtb.innerHTML = uv.map(v => '<tr><td><span class="severity-badge severity-' + v.severity + '">' + v.severity + '</span></td><td><strong>' + v.label + '</strong></td><td>' + v.class + '</td><td>' + (v.confidence * 100).toFixed(1) + '%</td><td>' + formatTime(v.timestamp_sec) + '</td></tr>').join('');
        }
        const ta = $('#timelineArea');
        const fd = data.frame_detections || [];
        if (fd.length === 0) { ta.innerHTML = '<div class="no-results">No frame detections</div>'; }
        else {
            const entries = fd.slice(0, 50);
            ta.innerHTML = entries.map(f => {
                const tags = f.detections.map(d => '<span class="timeline-tag ' + (isViolationClass(d.class) ? 'timeline-tag-violation' : 'timeline-tag-equip') + '">' + d.class + ' ' + (d.confidence * 100).toFixed(0) + '%</span>').join('');
                return '<div class="timeline-entry"><span class="timeline-time">' + formatTime(f.timestamp) + '</span><div class="timeline-detections">' + tags + '</div></div>';
            }).join('');
            if (fd.length > 50) ta.innerHTML += '<div class="no-results">... and ' + (fd.length - 50) + ' more frames</div>';
        }
        videoResults.classList.remove('hidden');
    }
})();
