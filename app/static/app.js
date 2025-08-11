/* global Tone, Midi */
(() => {
    const log = (...a) => console.log("[app.js]", ...a);
    const qs = (sel, el = document) => el.querySelector(sel);
    const qsa = (sel, el = document) => Array.from(el.querySelectorAll(sel));

    let localPart = null;

    function ready(fn) {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", fn, { once: true });
        } else {
            fn();
        }
    }

    ready(() => {
        log("loaded");
        const form = qs("#uploadForm");
        const resultEl = qs("#result");
        const submitBtn = qs("#submitBtn");
        const topKInput = qs("#top_k");
        const fileinput = qs("#file");
        const samplesEl = qs("#samples");

        if (form) {
            // When user selects a local file:
            document.getElementById("file").addEventListener("change", async (e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                const buf = await f.arrayBuffer();
                await parseMidi(buf);
            });

            // When user clicks a sample button:
            document.addEventListener("click", async (e) => {
                const btn = e.target.closest(".btn-predict");
                if (!btn) return;
                const url = btn.dataset.url;
                const res = await fetch(url);
                const buf = await res.arrayBuffer();
                await parseMidi(buf); // enable playback
                // you already post the same data to /predict elsewhere
            });
        }

        // Form submit -> /predict
        if (form && resultEl) {
            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                showInfo(resultEl, "Uploading and predicting…");
                disable(submitBtn, true);
                try {
                    const fd = new FormData(form);
                    const res = await fetch(form.action, { method: "POST", body: fd });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok)
                        return showError(resultEl, data.error || `HTTP ${res.status}`);
                    updateResults(resultEl, data);
                } catch (err) {
                    showError(resultEl, String(err));
                } finally {
                    disable(submitBtn, false);
                }
            });
        } else {
            log("form or result container not found; skipping form binding");
        }

        // Sample buttons — event delegation
        if (samplesEl && resultEl) {
            samplesEl.addEventListener("click", async (e) => {
                const btn = e.target.closest("button");
                if (!btn) return;
                const url = btn.dataset.url;
                if (!url) return;
                if (btn.classList.contains("btn-predict")) {
                    return predictSample(
                        url,
                        resultEl,
                        submitBtn,
                        Number(topKInput?.value || 4)
                    );
                }
            });

            // Also bind directly in case delegation misses due to markup nesting
            qsa(".btn-predict", samplesEl).forEach((b) =>
                b.addEventListener("click", () =>
                    predictSample(
                        b.dataset.url,
                        resultEl,
                        submitBtn,
                        Number(topKInput?.value || 4)
                    )
                )
            );
        } else {
            log("no #samples container found; skipping sample bindings");
        }
    });

    // ---------- UI helpers ----------
    function disable(el, v) {
        if (el) el.disabled = !!v;
    }
    function ensureResultsVisible(el) {
        if (el) el.style.display = "block";
    }
    function showInfo(el, msg) {
        ensureResultsVisible(el);
        el.innerHTML = `<div class="ok">${escapeHtml(msg)}</div>`;
    }
    function showError(el, msg) {
        ensureResultsVisible(el);
        el.innerHTML = `<div class="error"><strong>Error:</strong> ${escapeHtml(
            msg
        )}</div>`;
    }

    function updateResults(resultEl, data) {
        ensureResultsVisible(resultEl);

        const topList = (data.top || [])
            .map(
                (t) =>
                    `<li><strong>${escapeHtml(t.label)}</strong> — ${(
                        t.prob * 100
                    ).toFixed(2)}%</li>`
            )
            .join("");

        const zipped = (data.classes || [])
            .map((label, i) => ({
                label,
                prob: data.probs ? data.probs[i] : 0,
                i,
            }))
            .sort((a, b) => b.prob - a.prob);

        const rows = zipped
            .map(
                (z) => `
        <tr>
          <td>${escapeHtml(z.label)}</td>
          <td>${(z.prob * 100).toFixed(2)}%</td>
          <td><div class="bar"><span style="width:${(z.prob * 100).toFixed(
                    2
                )}%"></span></div></td>
        </tr>
      `
            )
            .join("");

        resultEl.innerHTML = `
        <div class="ok">
          <div><strong>Chunks processed:</strong> ${data.chunks}</div>
          <div><strong>Aggregation:</strong> ${escapeHtml(
            data.aggregation || "mean"
        )}</div>
        </div>
        <div class="ok">
          <h3 style="margin:0 0 8px;">Top predictions</h3>
          <ul>${topList}</ul>
        </div>
        <div class="ok">
          <h3 style="margin:0 0 8px;">All class probabilities</h3>
          <table>
            <thead><tr><th>Class</th><th>Prob</th><th>Chart</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    // ---------- Predict helpers ----------
    async function predictSample(url, resultEl, submitBtn, topK = 4) {
        try {
            showInfo(resultEl, "Loading sample and predicting…");
            disable(submitBtn, true);

            const resFile = await fetch(url, { cache: "no-cache" });
            if (!resFile.ok)
                throw new Error(`Failed to load sample: ${resFile.status}`);
            const blob = await resFile.blob();

            try {
                await Tone.start();                    
                const ab = await blob.arrayBuffer();
                const midi = new Midi(ab);         
                await parseMidi(ab); // enable playback
              } catch (e) {
                console.warn('Playback failed:', e);
              }

            const name = url.split("/").pop() || "sample.mid";
            const file = new File([blob], name, { type: blob.type || "audio/midi" });

            const fd = new FormData();
            fd.append("file", file);
            fd.append("top_k", String(topK));

            const res = await fetch("/predict", { method: "POST", body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok)
                return showError(resultEl, data.error || `HTTP ${res.status}`);
            updateResults(resultEl, data);
        } catch (e) {
            showError(resultEl, e.message || String(e));
        } finally {
            disable(submitBtn, false);
        }
    }

    // === Tone.js + MIDI playback state ===
    let midiData = null;
    let parts = [];
    let isLoadedForPlayback = false;
    let lastWavUrl = null;

    function revokeLastUrl() {
        if (lastWavUrl) {
            URL.revokeObjectURL(lastWavUrl);
            lastWavUrl = null;
        }
    }
    const fileInput = qs("#file");
    const playBtn  = qs('#playBtn');
    const pauseBtn = qs('#pauseBtn');
    const stopBtn  = qs('#stopBtn');
    const exportBtn = document.getElementById("exportBtn");
    const exportProgress = document.getElementById('exportProgress');
    const downloadLink = document.getElementById("downloadLink");
    const downloadBtn = document.getElementById("downloadBtn");

    const channelSynths = Array.from({ length: 16 }, () =>
        new Tone.PolySynth(Tone.Synth, { maxPolyphony: 64 }).toDestination()
    );

    channelSynths.forEach((s) =>
        s.set({
            envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 0.12 },
        })
    );

    // Simple percussive tweak for channel 9
    channelSynths[9].set({
        envelope: { attack: 0.001, decay: 0.05, sustain: 0.001, release: 0.05 },
        oscillator: { type: "square" },
    });

    const analyser = Tone.getContext().rawContext.createAnalyser();
    analyser.fftSize = 2048;
    const freqBins = new Uint8Array(analyser.frequencyBinCount);
    const timeBins = new Uint8Array(analyser.fftSize);

    Tone.Destination._internalChannels[0].connect(analyser);

    // Optional local play for uploaded file
    // const playLocalBtn = qs('#playLocalBtn');
    // const stopLocalBtn = qs('#stopLocalBtn');

    if (playBtn && fileInput) {
        playBtn.addEventListener('click', async () => {
        //   const f = fileInput.files?.[0];
        //   if (!f) return alert('Choose a MIDI file first or click a sample.');
          await Tone.start(); Tone.Transport.start('+0.05');
        });
      }

      if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
          try { Tone.Transport.pause(); } catch {}
        });
      }

      if (stopBtn) {
        stopBtn.addEventListener('click', () => {
          stopLocal();           
        });
      }

    // if (playBtn) playBtn.addEventListener('click', startPlayback);
    // pauseBtn.addEventListener("click", pausePlayback);
    // stopBtn.addEventListener("click", stopPlayback);

    downloadBtn.addEventListener("click", () => {
        if (!downloadLink.href) return;
        // Programmatically click the hidden anchor so the browser downloads the file
        downloadLink.click();
    });

    exportBtn.addEventListener('click', async () => {
        if (!midiData) return;

        // UI: start
        startExportUI();

        // Faux progress ticker based on renderSeconds
        let progressTimer;
        try {
            const renderSeconds = Math.ceil(midiData.duration + 0.5);
            const startTs = performance.now();
            progressTimer = setInterval(() => {
                const elapsed = (performance.now() - startTs) / 1000;
                const pct = Math.min(99, Math.floor((elapsed / renderSeconds) * 100)); // cap at 99 until done
                updateExportPercent(pct);
            }, 120);

            // Do the offline render
            const buffer = await Tone.Offline(({ transport }) => {
                const synths = Array.from({ length: 16 }, () => new Tone.PolySynth(Tone.Synth, { maxPolyphony: 64 }).toDestination());
                synths[9].set({ envelope: { attack: 0.001, decay: 0.05, sustain: 0.001, release: 0.05 }, oscillator: { type: 'square' } });

                // schedule new Parts here too
                midiData.tracks.forEach(track => {
                    if (!track.notes?.length) return;
                    const ch = track.channel ?? 0;

                    const events = track.notes.map(n => ({
                    time: n.time,
                    name: n.name,
                    duration: n.duration,
                    velocity: n.velocity ?? 0.8
                    }));

                    const part = new Tone.Part((time, note) => {
                    synths[ch].triggerAttackRelease(note.name, note.duration, time, note.velocity);
                    }, events);

                    part.start(0);
                });

                transport.bpm.value = midiData.header.tempos?.[0]?.bpm ?? 120;
                transport.start(0);
            }, Math.ceil(midiData.duration + 0.5));

            // Encode + prepare download
            const wavBlob = encodeWav(buffer);
            const url = URL.createObjectURL(wavBlob);
            downloadLink.href = url;
            downloadLink.download = `${(midiData.header.name || 'render')}.wav`;

            // UI: finish
            clearInterval(progressTimer);
            updateExportPercent(100);
            setTimeout(() => endExportUI(), 200); // brief 100% flash
            showDownloadReady();

        } catch (err) {
            console.error(err);
            clearInterval(progressTimer);
            failExportUI('Error');
        }
    });

    function startExportUI() {
        exportBtn.classList.add('is-loading');
        exportBtn.disabled = true;
        if (exportProgress) {
            exportProgress.style.display = 'inline';
            exportProgress.textContent = 'Rendering…';
        }
        downloadBtn.style.display = 'none';
    }

    function updateExportPercent(pct) {
        if (!exportProgress) return;
        exportProgress.textContent = `Rendering ${pct}%`;
    }

    function endExportUI() {
        exportBtn.classList.remove('is-loading');
        exportBtn.disabled = false;
        if (exportProgress) exportProgress.style.display = 'none';
    }

    function failExportUI(msg = 'Failed') {
        exportBtn.classList.remove('is-loading');
        exportBtn.removeAttribute('aria-busy');
        exportBtn.disabled = false;
        if (exportProgress) {
            exportProgress.style.display = 'inline';
            exportProgress.textContent = msg;
            setTimeout(() => { exportProgress.style.display = 'none'; }, 2000);
        }
    }

    // Call this when export completes successfully to reveal the Download button
    function showDownloadReady() {
        downloadBtn.style.display = 'inline-block';
    }

    // ===== Visualizer =====
    const vizCanvas = document.getElementById('viz');
    const vctx = vizCanvas.getContext('2d', { alpha: false });

    function resizeViz() {
        const dpr = Math.max(1, window.devicePixelRatio || 1);
      
        // Use the CSS size (from the styles above), not the intrinsic attrs
        const cssW = vizCanvas.clientWidth || 600;
        const cssH = vizCanvas.clientHeight || 180;
      
        // Set the backing store to DPR-scaled pixels
        vizCanvas.width  = Math.round(cssW * dpr);
        vizCanvas.height = Math.round(cssH * dpr);
      
        // Reset then scale the drawing context to map 1 unit = 1 CSS px
        vctx.setTransform(1, 0, 0, 1, 0, 0);
        vctx.scale(dpr, dpr);
      }
      window.addEventListener('resize', resizeViz);
      resizeViz();


    // Modes: "bars" | "radial" | "wave"
    let vizMode = 'bars';
    document.querySelectorAll('.viz-btn').forEach(b => {
        b.addEventListener('click', () => {
            document.querySelectorAll('.viz-btn').forEach(x => x.classList.remove('active'));
            b.classList.add('active');
            vizMode = b.dataset.mode;
        });
    });
    document.querySelector('.viz-btn[data-mode="bars"]')?.classList.add('active');

    // Gradient helper
    function makeGradient() {
        const g = vctx.createLinearGradient(0, 0, vizCanvas.width, 0);
        g.addColorStop(0, '#4a90e2');
        g.addColorStop(0.5, '#50c9c3');
        g.addColorStop(1, '#4a90e2');
        return g;
    }
    let grad = makeGradient();

    // Simple peak caps for bar mode
    let peaks = new Float32Array(128).fill(0);
    let peakVel = new Float32Array(128).fill(0);

    // Particle bursts on note events
    const particles = [];
    function spawnBurst(x, y, power = 1) {
        for (let i = 0; i < 12; i++) {
            particles.push({
                x, y,
                vx: (Math.random() - 0.5) * 6 * power,
                vy: -Math.random() * 5 * power - 1,
                life: 0.9 + Math.random() * 0.6,
                age: 0,
                size: 2 + Math.random() * 2
            });
        }
    }
    function stepParticles(dt) {
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.age += dt;
            p.x += p.vx * dt * 60;
            p.y += p.vy * dt * 60;
            p.vy += 0.08 * dt * 60; // gravity
            if (p.age > p.life) particles.splice(i, 1);
        }
    }
    function drawParticles() {
        for (const p of particles) {
            const t = 1 - p.age / p.life;
            vctx.globalAlpha = Math.max(0, t);
            vctx.fillStyle = '#ffffff';
            vctx.fillRect(p.x, p.y, p.size, p.size);
        }
        vctx.globalAlpha = 1;
    }

    // Main draw loop
    let lastTs = performance.now();
    function drawViz() {
        requestAnimationFrame(drawViz);
        const now = performance.now();
        const dt = Math.min(0.1, (now - lastTs) / 1000);
        lastTs = now;

        if (vizMode === 'wave') {
            analyser.getByteTimeDomainData(timeBins);
        } else {
            analyser.getByteFrequencyData(freqBins);
        }

        // Clear
        vctx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);

        const W = vizCanvas.clientWidth;
        const H = vizCanvas.clientHeight;

        // paint background to match the card
        vctx.fillStyle = getComputedStyle(document.documentElement)
        .getPropertyValue('--card') || '#141b2a';
        vctx.fillRect(0, 0, W, H);

        if (vizMode === 'bars') {
            const bars = 128;
            const step = Math.floor(freqBins.length / bars);
            const bw = W / bars;
            if (vizCanvas.width !== 0) grad = makeGradient();
            vctx.fillStyle = grad;

            for (let i = 0; i < bars; i++) {
                const v = freqBins[i * step] / 255;
                const h = Math.pow(v, 0.8) * H; // slight compand
                const x = i * bw;
                // bar
                vctx.fillRect(x + 1, H - h, bw - 2, h);

                // peak cap
                const target = H - h;
                if (peaks[i] === 0) peaks[i] = target;
                peaks[i] = Math.min(peaks[i] + peakVel[i], target);
                peakVel[i] += 0.6;        // gravity for cap
                if (target < peaks[i]) {  // hit new higher bar, reset velocity
                    peaks[i] = target;
                    peakVel[i] = -8;
                }
                vctx.fillStyle = '#fff';
                vctx.fillRect(x + 1, peaks[i] - 3, bw - 2, 3);
                vctx.fillStyle = grad;
            }
        }

        if (vizMode === 'radial') {
            const bars = 120;
            const step = Math.floor(freqBins.length / bars);
            const cx = W / 2;
            const cy = H / 2;
            const baseR = Math.min(W, H) * 0.25;
            if (vizCanvas.width !== 0) grad = makeGradient();
            vctx.strokeStyle = grad;
            vctx.lineWidth = 2;

            for (let i = 0; i < bars; i++) {
                const v = freqBins[i * step] / 255;
                const len = baseR + Math.pow(v, 0.8) * (Math.min(W, H) * 0.35);
                const a = (i / bars) * Math.PI * 2;
                const x1 = cx + Math.cos(a) * baseR;
                const y1 = cy + Math.sin(a) * baseR;
                const x2 = cx + Math.cos(a) * len;
                const y2 = cy + Math.sin(a) * len;
                vctx.beginPath();
                vctx.moveTo(x1, y1);
                vctx.lineTo(x2, y2);
                vctx.stroke();
            }

            // glow circle
            vctx.beginPath();
            vctx.arc(cx, cy, baseR, 0, Math.PI * 2);
            vctx.strokeStyle = 'rgba(255,255,255,.25)';
            vctx.lineWidth = 1;
            vctx.stroke();
        }

        if (vizMode === 'wave') {
            vctx.beginPath();
            const mid = H * 0.5;
            for (let i = 0; i < timeBins.length; i++) {
                const x = (i / (timeBins.length - 1)) * W;
                const y = mid + ((timeBins[i] - 128) / 128) * (H * 0.45);
                if (i === 0) vctx.moveTo(x, y);
                else vctx.lineTo(x, y);
            }
            vctx.strokeStyle = 'rgba(255,255,255,0.9)';
            vctx.lineWidth = 2;
            vctx.stroke();

            // soft fill
            const gradY = vctx.createLinearGradient(0, 0, 0, H);
            gradY.addColorStop(0, 'rgba(80,201,195,0.25)');
            gradY.addColorStop(1, 'rgba(74,144,226,0.05)');
            vctx.lineTo(W, H);
            vctx.lineTo(0, H);
            vctx.closePath();
            vctx.fillStyle = gradY;
            vctx.fill();
        }

        // particles layer
        stepParticles(dt);
        drawParticles();
    }
    drawViz();

    // Parse a MIDI ArrayBuffer into Midi object
    async function parseMidi(arrayBuffer) {
        midiData = new Midi(arrayBuffer);
        isLoadedForPlayback = false;
        // Clear any previous parts
        parts.forEach((p) => p.dispose());
        parts = [];

        // one Part per track
        midiData.tracks.forEach(track => {
            if (!track.notes?.length) return;
            const channel = track.channel ?? 0;

            // optional: deoverlap and clamp durations if you need
            const events = track.notes.map(n => ({
                time: n.time,
                name: n.name,
                duration: n.duration,
                velocity: n.velocity ?? 0.8
            }));

            const part = new Tone.Part((time, note) => {
                channelSynths[channel].triggerAttackRelease(
                    note.name, note.duration, time, note.velocity
                );

                // hook your visualizer particle here if you added it
                const midiNum = Tone.Frequency(note.name).toMidi();
                spawnBurst((midiNum % 88) / 88 * vizCanvas.width, vizCanvas.height * 0.6, 0.8);
            }, events);

            part.start(0);        // schedule relative to Transport position 0
            parts.push(part);
        });

       // set tempo and auto-stop
        const bpm = midiData.header.tempos?.[0]?.bpm ?? 120;
        Tone.Transport.bpm.value = bpm;
        const endTime = midiData.duration;
        Tone.Transport.scheduleOnce(() => {
            Tone.Transport.stop();
            Tone.Transport.position = 0;
        }, endTime + 0.1);

        
        isLoadedForPlayback = true;

        revokeLastUrl();
        downloadBtn.style.display = "none";
        downloadLink.removeAttribute("href");
        downloadLink.removeAttribute("download");
    }

    function deoverlapAndClamp(notes, { maxDur = 2.5, gap = 0.005, minDur = 0.03 } = {}) {
        const byPitch = new Map();
        const out = [];
        for (const n of notes) {
            const key = n.midi; // numeric pitch
            const prev = byPitch.get(key);

            // If previous same-pitch note overlaps this start, shorten the previous a hair
            if (prev && prev.time + prev.duration > n.time - gap) {
                prev.duration = Math.max(minDur, n.time - prev.time - gap);
            }

            const dur = Math.max(minDur, Math.min(maxDur, n.duration));
            const nn = { ...n, duration: dur };
            byPitch.set(key, nn);
            out.push(nn);
        }
        return out;
    }

    // Control helpers
    async function ensureContextStarted() {
        if (Tone.context.state !== "running") {
            await Tone.start();
        }
    }

    // async function startPlayback() { await Tone.start(); Tone.Transport.start('+0.05'); }
    // function pausePlayback() { Tone.Transport.pause(); }
    // function stopPlayback() { Tone.Transport.stop(); Tone.Transport.position = 0; }

    // Tiny WAV encoder for an AudioBuffer
    function encodeWav(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const samples = audioBuffer.length;
        const bytesPerSample = 2;
        const blockAlign = numChannels * bytesPerSample;
        const buffer = new ArrayBuffer(44 + samples * blockAlign);
        const view = new DataView(buffer);

        // RIFF header
        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + samples * blockAlign, true);
        writeString(view, 8, "WAVE");

        // fmt chunk
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 8 * bytesPerSample, true);

        // data chunk
        writeString(view, 36, "data");
        view.setUint32(40, samples * blockAlign, true);

        // interleave
        const channels = Array.from({ length: numChannels }, (_, i) =>
            audioBuffer.getChannelData(i)
        );
        let offset = 44;
        for (let i = 0; i < samples; i++) {
            for (let c = 0; c < numChannels; c++) {
                // clamp & scale float [-1,1] to int16
                let s = Math.max(-1, Math.min(1, channels[c][i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                offset += 2;
            }
        }
        return new Blob([view], { type: "audio/wav" });

        function writeString(dv, pos, str) {
            for (let i = 0; i < str.length; i++)
                dv.setUint8(pos + i, str.charCodeAt(i));
        }
    }

    function stopLocal() {
        try {
            Tone.Transport.stop();
        } catch (e) { }
        if (localPart) {
            localPart.dispose();
            localPart = null;
        }
    }

    // ---------- Utils ----------
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    // expose stop if you added a Stop button
    window._stopLocalPlayback = stopLocal;
})();
