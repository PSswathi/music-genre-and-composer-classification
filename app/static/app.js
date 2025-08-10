/* global Tone, Midi */
(() => {
    const log = (...a) => console.log('[app.js]', ...a);
    const qs  = (sel, el = document) => el.querySelector(sel);
    const qsa = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  
    let localPart = null;
  
    function ready(fn) {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fn, { once: true });
      } else {
        fn();
      }
    }
  
    ready(() => {
      log('loaded');
      const form      = qs('#uploadForm');
      const resultEl  = qs('#result');
      const submitBtn = qs('#submitBtn');
      const topKInput = qs('#top_k');
      const fileInput = qs('#file');
      const samplesEl = qs('#samples');
  
      // Form submit -> /predict
      if (form && resultEl) {
        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          showInfo(resultEl, 'Uploading and predicting…');
          disable(submitBtn, true);
          try {
            const fd = new FormData(form);
            const res = await fetch(form.action, { method: 'POST', body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) return showError(resultEl, data.error || `HTTP ${res.status}`);
            updateResults(resultEl, data);
          } catch (err) {
            showError(resultEl, String(err));
          } finally {
            disable(submitBtn, false);
          }
        });
      } else {
        log('form or result container not found; skipping form binding');
      }
  
      // Sample buttons — event delegation
      if (samplesEl && resultEl) {
        samplesEl.addEventListener('click', async (e) => {
          const btn = e.target.closest('button');
          if (!btn) return;
          const url = btn.dataset.url;
          if (!url) return;
          if (btn.classList.contains('btn-predict')) {
            return predictSample(url, resultEl, submitBtn, Number(topKInput?.value || 4));
          }
        });
  
        // Also bind directly in case delegation misses due to markup nesting
        qsa('.btn-predict', samplesEl).forEach(b => b.addEventListener('click', () =>
          predictSample(b.dataset.url, resultEl, submitBtn, Number(topKInput?.value || 4))
        ));
      } else {
        log('no #samples container found; skipping sample bindings');
      }
  
      // Optional local play for uploaded file
      const playLocalBtn = qs('#playLocalBtn');
      const stopLocalBtn = qs('#stopLocalBtn');
      if (playLocalBtn && fileInput) {
        playLocalBtn.addEventListener('click', async () => {
          const f = fileInput.files?.[0];
          if (!f) return alert('Choose a MIDI file first.');
          await playLocal(f);
          disable(stopLocalBtn, false);
        });
      }
      if (stopLocalBtn) {
        stopLocalBtn.addEventListener('click', () => {
          stopLocal();
          disable(stopLocalBtn, true);
        });
      }
    });
  
    // ---------- UI helpers ----------
    function disable(el, v) { if (el) el.disabled = !!v; }
    function ensureResultsVisible(el) { if (el) el.style.display = 'block'; }
    function showInfo(el, msg) {
      ensureResultsVisible(el);
      el.innerHTML = `<div class="ok">${escapeHtml(msg)}</div>`;
    }
    function showError(el, msg) {
      ensureResultsVisible(el);
      el.innerHTML = `<div class="error"><strong>Error:</strong> ${escapeHtml(msg)}</div>`;
    }
  
    function updateResults(resultEl, data) {
      ensureResultsVisible(resultEl);
      const hasAudio = !!data.audio_data_url;
      const audioBlock = hasAudio
        ? `<div class="ok">
             <h3 style="margin:0 0 8px;">Preview</h3>
             <audio id="audioPreview" controls src="${data.audio_data_url}" preload="none"></audio>
             <canvas id="audioViz"></canvas>
           </div>`
        : `<div class="ok">
             <h3 style="margin:0 0 8px;">Preview</h3>
             <div class="muted">Audio preview not available.</div>
           </div>`;
  
      const topList = (data.top || []).map((t) =>
        `<li><strong>${escapeHtml(t.label)}</strong> — ${(t.prob * 100).toFixed(2)}%</li>`
      ).join('');
  
      const zipped = (data.classes || []).map((label, i) => ({
        label, prob: data.probs ? data.probs[i] : 0, i
      })).sort((a,b) => b.prob - a.prob);
  
      const rows = zipped.map((z) => `
        <tr>
          <td>${escapeHtml(z.label)}</td>
          <td>${(z.prob * 100).toFixed(2)}%</td>
          <td><div class="bar"><span style="width:${(z.prob * 100).toFixed(2)}%"></span></div></td>
        </tr>
      `).join('');
  
      resultEl.innerHTML = `
        <div class="ok">
          <div><strong>Chunks processed:</strong> ${data.chunks}</div>
          <div><strong>Aggregation:</strong> ${escapeHtml(data.aggregation || 'mean')}</div>
        </div>
        ${audioBlock}
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
  
      const audioEl  = document.getElementById('audioPreview');
      const canvasEl = document.getElementById('audioViz');
      if (audioEl && canvasEl) setupAudioVisualizer(audioEl, canvasEl);
    }
  
    // ---------- Predict helpers ----------
    async function predictSample(url, resultEl, submitBtn, topK = 4) {
      try {
        showInfo(resultEl, 'Loading sample and predicting…');
        disable(submitBtn, true);
  
        const resFile = await fetch(url, { cache: 'no-cache' });
        if (!resFile.ok) throw new Error(`Failed to load sample: ${resFile.status}`);
        const blob = await resFile.blob();
        const name = url.split('/').pop() || 'sample.mid';
        const file = new File([blob], name, { type: blob.type || 'audio/midi' });
  
        const fd = new FormData();
        fd.append('file', file);
        fd.append('top_k', String(topK));
  
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) return showError(resultEl, data.error || `HTTP ${res.status}`);
        updateResults(resultEl, data);
      } catch (e) {
        showError(resultEl, e.message || String(e));
      } finally {
        disable(submitBtn, false);
      }
    }
  
    // ---------- Local playback ----------
    async function playLocal(file) {
      if (!file) return alert('Choose a MIDI file first.');
      if (!(window.Tone && window.Midi)) return alert('Client-side playback libs not loaded.');
      await Tone.start();
      const arrayBuffer = await file.arrayBuffer();
      const midi = new Midi(arrayBuffer);
      startTonePlayback(midi);
    }
  
    async function playSample(url) {
      if (!(window.Tone && window.Midi)) return alert('Client-side playback libs not loaded.');
      await Tone.start();
      const res = await fetch(url, { cache: 'no-cache' });
      if (!res.ok) return alert('Failed to load sample.');
      const ab = await res.arrayBuffer();
      const midi = new Midi(ab);
      startTonePlayback(midi);
    }
  
    function startTonePlayback(midi) {
      const now = Tone.now() + 0.2;
      const vol = new Tone.Volume(-6).toDestination();
  
      try { Tone.Transport.stop(); } catch (e) {}
      Tone.Transport.position = 0;
  
      if (localPart) { localPart.dispose(); localPart = null; }
  
      const synths = midi.tracks.map(t =>
        t.channel === 9
          ? new Tone.MembraneSynth().connect(vol)
          : new Tone.PolySynth(Tone.Synth).connect(vol)
      );
  
      const events = [];
      midi.tracks.forEach((track, ti) => {
        track.notes.forEach(n => {
          events.push({ time: n.time, dur: n.duration, pitch: n.name, ti });
        });
      });
  
      localPart = new Tone.Part((time, ev) => {
        synths[ev.ti].triggerAttackRelease(ev.pitch, ev.dur, time);
      }, events).start(now);
  
      Tone.Transport.start();
    }
  
    function stopLocal() {
      try { Tone.Transport.stop(); } catch (e) {}
      if (localPart) { localPart.dispose(); localPart = null; }
    }
  
    // ---------- Visualizer ----------
    function setupAudioVisualizer(audioEl, canvasEl) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        console.warn('Web Audio API not supported');
        return;
      }
      const ctx = new AudioCtx();
      const src = ctx.createMediaElementSource(audioEl);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      analyser.connect(ctx.destination);
  
      const c = canvasEl.getContext('2d');
  
      function resize() {
        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvasEl.clientWidth || 700;
        const cssHeight = canvasEl.clientHeight || 140;
        canvasEl.width = Math.floor(cssWidth * dpr);
        canvasEl.height = Math.floor(cssHeight * dpr);
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      resize();
      window.addEventListener('resize', resize);
  
      let rafId = null;
      const bars = 64;
      const data = new Uint8Array(analyser.frequencyBinCount);
  
      function draw() {
        rafId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(data);
  
        const width = canvasEl.width / (window.devicePixelRatio || 1);
        const height = canvasEl.height / (window.devicePixelRatio || 1);
        c.clearRect(0, 0, width, height);
  
        const step = Math.max(1, Math.floor(data.length / bars));
        const barWidth = width / bars;
        const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#22c55e';
  
        for (let i = 0; i < bars; i++) {
          const v = data[i * step] / 255;
          const h = v * height;
          const x = i * barWidth;
          const grad = c.createLinearGradient(0, height - h, 0, height);
          grad.addColorStop(0, accent.trim());
          grad.addColorStop(1, '#1f2937');
          c.fillStyle = grad;
          c.fillRect(x + 1, height - h, barWidth - 2, h);
        }
      }
  
      function start() {
        if (ctx.state === 'suspended') ctx.resume();
        draw();
      }
      function stop() {
        // stop drawing on pause/ended
        // We keep ctx to avoid interruptions on resume
      }
  
      audioEl.addEventListener('play', start);
      audioEl.addEventListener('pause', stop);
      audioEl.addEventListener('ended', stop);
    }
  
    // ---------- Utils ----------
    function escapeHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }
  
    // expose stop if you added a Stop button
    window._stopLocalPlayback = stopLocal;
  })();
  