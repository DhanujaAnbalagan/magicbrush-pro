// ─── Config ────────────────────────────────────────────────────
const API_URL = "https://luetically-aplanatic-jackson.ngrok-free.dev";
const HF_API  = "https://datasets-server.huggingface.co/rows?dataset=osunlp%2FMagicBrush&config=default&split=train&offset=0&limit=100";

const { useState, useRef, useEffect, useCallback } = React;

// ─── Toast System ────────────────────────────────────────────────
function useToast() {
    const [toasts, setToasts] = useState([]);
    const add = useCallback((msg, type = 'info', duration = 4000) => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, msg, type }]);
        setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration);
    }, []);
    return { toasts, toast: add };
}

function ToastContainer({ toasts }) {
    return (
        <div className="toast-container">
            {toasts.map(t => (
                <div key={t.id} className={`toast toast-${t.type}`}>
                    {t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : t.type === 'warn' ? '⚠️' : 'ℹ️'} {t.msg}
                </div>
            ))}
        </div>
    );
}

// ─── Before/After Slider ─────────────────────────────────────────
function BeforeAfter({ original, generated }) {
    const [split, setSplit] = useState(50);
    return (
        <div className="before-after-wrap">
            <div className="before-after-inner" style={{ '--split': `${split}%` }}>
                <img className="ba-img ba-before" src={original} alt="Original" />
                <img className="ba-img ba-after"  src={generated} alt="Generated" style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
                <div className="ba-divider" style={{ left: `${split}%` }}>
                    <div className="ba-handle">◀▶</div>
                </div>
            </div>
            <input type="range" className="ba-slider" min="0" max="100"
                value={split} onChange={e => setSplit(Number(e.target.value))} />
            <div className="ba-labels">
                <span>Original</span><span>Generated</span>
            </div>
        </div>
    );
}

// ─── Fullscreen Lightbox ──────────────────────────────────────────
function Lightbox({ result, original, onClose }) {
    const [showBA, setShowBA] = useState(false);
    useEffect(() => {
        const handler = e => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);
    return (
        <div className="lightbox-overlay" onClick={onClose}>
            <div className="lightbox-box" onClick={e => e.stopPropagation()}>
                <div className="lightbox-toolbar">
                    <button className={`lb-btn ${showBA ? 'active' : ''}`} onClick={() => setShowBA(v => !v)}>
                        ◀▶ Before / After
                    </button>
                    <div className="lb-scores">
                        <span>CLIP {result.clip_score}</span>
                        <span>LPIPS {result.lpips_score}</span>
                        <span className="lb-final">Score {result.final_score}</span>
                    </div>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>
                {showBA
                    ? <BeforeAfter original={original} generated={result.image} />
                    : <img className="lightbox-img" src={result.image} alt="Result" />
                }
            </div>
        </div>
    );
}

// ─── Main App ───────────────────────────────────────────────────
function App() {
    const [image, setImage]                   = useState(null);
    const [prompt, setPrompt]                 = useState("");
    const [isGenerating, setIsGenerating]     = useState(false);
    const [results, setResults]               = useState([]);
    const [selectedId, setSelectedId]         = useState(null);
    const maskCanvasRef                       = useRef(null);
    const { toasts, toast }                   = useToast();

    // Settings
    const [settingsOpen, setSettingsOpen]     = useState(false);
    const [guidance, setGuidance]             = useState(7.5);
    const [imageGuidance, setImageGuidance]   = useState(1.5);
    const [numSamples, setNumSamples]         = useState(4);
    const [inferenceSteps, setInferenceSteps] = useState(30);
    const [negativePrompt, setNegativePrompt] = useState("blurry, deformed, ugly, bad anatomy, low quality, watermark");

    // Dataset modal
    const [modalOpen, setModalOpen]           = useState(false);
    const [allSamples, setAllSamples]         = useState([]);
    const [searchQuery, setSearchQuery]       = useState("");
    const [loadingSamples, setLoadingSamples] = useState(false);
    const [modalFetched, setModalFetched]     = useState(false);

    // Lightbox
    const [lightboxResult, setLightboxResult] = useState(null);

    const filteredSamples = searchQuery.trim()
        ? allSamples.filter(s => s.prompt.toLowerCase().includes(searchQuery.toLowerCase()))
        : allSamples;

    const selectedResult = results.find(r => r.id === selectedId);

    // ── Dataset modal ─────────────────────────────────────────────
    const openModal = async () => {
        setModalOpen(true); setSearchQuery("");
        if (modalFetched) return;
        setLoadingSamples(true);
        try {
            const res  = await fetch(HF_API);
            const data = await res.json();
            if (data.rows) {
                setAllSamples(data.rows.map(r => ({ src: r.row.source_img?.src, prompt: r.row.instruction })).filter(s => s.src && s.prompt));
            }
        } catch(e) { toast("Could not load dataset from HuggingFace", 'error'); }
        finally    { setLoadingSamples(false); setModalFetched(true); }
    };

    const loadSample = async (sample) => {
        setModalOpen(false); setResults([]); setSelectedId(null); setPrompt(sample.prompt);
        try {
            const res  = await fetch(sample.src);
            const blob = await res.blob();
            const b64  = await new Promise(resolve => { const r = new FileReader(); r.onload = ev => resolve(ev.target.result); r.readAsDataURL(blob); });
            setImage(b64);
            toast("Sample loaded — paint your mask, then Generate!", 'success');
        } catch(e) { setImage(sample.src); }
    };

    // ── Generate ──────────────────────────────────────────────────
    const handleGenerate = useCallback(async () => {
        if (!image || !prompt.trim()) { toast("Upload an image and enter a prompt first", 'warn'); return; }
        setIsGenerating(true); setResults([]); setSelectedId(null);

        const hasMask = maskCanvasRef.current?.hasPaintedMask?.() || false;
        if (!hasMask) {
            const ok = window.confirm("⚠️ No mask painted!\n\nWithout a mask the ENTIRE image will be regenerated and you'll lose your original background.\n\nCancel to paint a mask, OK to continue anyway.");
            if (!ok) { setIsGenerating(false); return; }
        }

        const maskBase64 = maskCanvasRef.current?.getMaskBase64?.() || null;
        const formData   = new FormData();
        formData.append("image",                image);
        formData.append("prompt",               prompt.trim());
        formData.append("negative_prompt",      negativePrompt);
        formData.append("num_samples",          String(numSamples));
        formData.append("num_inference_steps",  String(inferenceSteps));
        if (hasMask && maskBase64) formData.append("mask", maskBase64);

        try {
            const resp = await fetch(`${API_URL}/generate`, { method: "POST", body: formData });
            if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);
            const data = await resp.json();
            if (data.samples?.length > 0) {
                const normalized = data.samples.map((s, i) => ({
                    id: i, image: s.image,
                    clip_score: s.scores.clip_score,
                    lpips_score: s.scores.lpips_score,
                    final_score: s.scores.final_score,
                }));
                setResults(normalized);
                setSelectedId(0); // auto-select best
                toast(`✨ ${normalized.length} candidates generated! Best is auto-selected.`, 'success');
            } else {
                toast("No results returned from backend", 'error');
            }
        } catch(err) {
            console.error(err);
            toast("Backend connection failed — is your Colab cell running? " + err.message, 'error', 7000);
        } finally { setIsGenerating(false); }
    }, [image, prompt, negativePrompt, numSamples, inferenceSteps]);

    // ── Copy to clipboard ─────────────────────────────────────────
    const copyToClipboard = async (result) => {
        try {
            const res  = await fetch(result.image);
            const blob = await res.blob();
            await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
            toast("Image copied to clipboard!", 'success');
        } catch(e) {
            toast("Copy failed — use Download instead", 'warn');
        }
    };

    // ── Keyboard shortcuts ────────────────────────────────────────
    useEffect(() => {
        const handler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleGenerate(); }
            if (e.key === 'Escape') { setLightboxResult(null); setModalOpen(false); }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [handleGenerate]);

    // ── Upload ────────────────────────────────────────────────────
    const setImgFromFile = (file) => {
        if (!file?.type?.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = ev => { setImage(ev.target.result); setResults([]); setSelectedId(null); toast("Image loaded — now paint your mask!", 'info'); };
        reader.readAsDataURL(file);
    };

    // ── Download ──────────────────────────────────────────────────
    const handleDownload = () => {
        if (selectedResult) {
            const a = document.createElement("a");
            a.href = selectedResult.image; a.download = `magicbrush-output-${selectedResult.id}.png`; a.click();
            toast("Download started!", 'success');
        }
    };

    const handleReset = () => { setImage(null); setResults([]); setSelectedId(null); setPrompt(""); toast("Canvas cleared — ready for a new image", 'info'); };

    return (
        <div className="app-container">
            <ToastContainer toasts={toasts} />
            {lightboxResult && <Lightbox result={lightboxResult} original={image} onClose={() => setLightboxResult(null)} />}

            {/* ── Header ── */}
            <header>
                <div className="logo-badge">✦ AI-Powered Editing</div>
                <h1>MagicBrush Pro</h1>
                <p>SD Inpainting · Mask-aware region editing · CLIP + LPIPS candidate ranking</p>
                <div className="shortcut-bar">
                    <kbd>Enter</kbd> Generate &nbsp;·&nbsp;
                    <kbd>Ctrl+Z</kbd> Undo mask &nbsp;·&nbsp;
                    <kbd>B</kbd> Brush &nbsp;·&nbsp;
                    <kbd>E</kbd> Erase &nbsp;·&nbsp;
                    <kbd>Esc</kbd> Close
                </div>
            </header>

            {/* ── Workspace ── */}
            <div className="workspace">
                {/* Left: Canvas */}
                <div className="left-panel">
                    <div className="panel-label">Source Image + Mask</div>
                    {!image ? (
                        <label className="upload-zone"
                            onDragOver={e => e.preventDefault()}
                            onDrop={e => { e.preventDefault(); setImgFromFile(e.dataTransfer.files[0]); }}>
                            <div className="upload-icon">📸</div>
                            <h2>Drop your image here</h2>
                            <p>or click to browse — PNG, JPG, WEBP</p>
                            <input type="file" accept="image/*" onChange={e => setImgFromFile(e.target.files[0])} />
                        </label>
                    ) : (
                        <MaskCanvas ref={maskCanvasRef} imageUrl={image} />
                    )}
                </div>

                {/* Right: Controls */}
                <div className="right-panel">

                    <button className="btn-dataset" onClick={openModal}>
                        <span style={{fontSize:'1.1rem'}}>📁</span>
                        <span>Browse MagicBrush Dataset</span>
                        <span className="badge-hot">100 samples</span>
                    </button>

                    <div className="divider" />

                    <div>
                        <div className="section-title">Edit Instruction</div>
                        <textarea className="prompt-box"
                            placeholder="e.g. make it winter · add a cat on the chair · change the sky to sunset..."
                            value={prompt} onChange={e => setPrompt(e.target.value)}
                            disabled={!image || isGenerating} rows={3} />
                    </div>

                    <button className={`btn-settings-toggle ${settingsOpen ? 'open' : ''}`}
                        onClick={() => setSettingsOpen(v => !v)}>
                        <span>⚙️ Generation Settings</span>
                        <span className="settings-arrow">{settingsOpen ? '▲' : '▼'}</span>
                    </button>

                    {settingsOpen && (
                        <div className="settings-panel">
                            <div>
                                <div className="section-title" style={{marginBottom:'0.4rem'}}>Negative Prompt</div>
                                <input className="neg-prompt-input" type="text" value={negativePrompt}
                                    onChange={e => setNegativePrompt(e.target.value)}
                                    placeholder="What to avoid in the output..." disabled={isGenerating} />
                            </div>
                            <div className="settings-grid">
                                {[
                                    { lbl:'🎲 Candidates',       val: numSamples,      set: setNumSamples,      min:1,  max:8,  step:1   },
                                    { lbl:'🔄 Inference Steps',  val: inferenceSteps,   set: setInferenceSteps,  min:10, max:80, step:5   },
                                    { lbl:'📝 Text Guidance',    val: guidance,         set: setGuidance,        min:1,  max:15, step:0.5 },
                                    { lbl:'🖼 Image Guidance',   val: imageGuidance,    set: setImageGuidance,   min:1,  max:3,  step:0.1 },
                                ].map(({ lbl, val, set, min, max, step }) => (
                                    <div key={lbl} className="slider-row">
                                        <div className="slider-header">
                                            <span className="slider-lbl">{lbl}</span>
                                            <span className="slider-val">{parseFloat(val).toFixed(step < 1 ? 1 : 0)}</span>
                                        </div>
                                        <input type="range" className="full" min={min} max={max} step={step}
                                            value={val} onChange={e => set(Number(e.target.value))} disabled={isGenerating} />
                                    </div>
                                ))}
                            </div>
                            <div className="settings-hint">⚡ Higher steps = better quality but slower</div>
                        </div>
                    )}

                    <div className="divider" />

                    <div className="status-hint"><div className="hint-dot" />GPU backend connected · ngrok tunnel active</div>

                    <button className="btn-generate" onClick={handleGenerate}
                        disabled={!image || !prompt.trim() || isGenerating}>
                        {isGenerating
                            ? <><div className="spinner-ring" style={{width:'20px',height:'20px',borderWidth:'2px'}} /> Generating…</>
                            : <>✨ Generate Edits &nbsp;<kbd style={{fontSize:'0.7rem',opacity:0.7,background:'rgba(255,255,255,0.15)',padding:'0.1rem 0.3rem',borderRadius:'3px'}}>Enter</kbd></>
                        }
                    </button>

                    {image && <button className="btn-secondary" onClick={handleReset} disabled={isGenerating}>↩ Start Over</button>}
                </div>
            </div>

            {/* ── Loading ── */}
            {isGenerating && (
                <div className="loading-section">
                    <div className="spinner-ring" />
                    <p>Generating {numSamples} candidate{numSamples !== 1 ? 's' : ''} · {inferenceSteps} steps · CLIP + LPIPS ranking…<br />
                    <span style={{fontSize:'0.8rem',opacity:0.7}}>Est. {Math.round(numSamples * inferenceSteps * 0.7)}–{Math.round(numSamples * inferenceSteps * 1.3)}s on GPU</span></p>
                    <div className="loading-steps">
                        {["SD Inpainting","Mask compositing","CLIP scoring","LPIPS scoring","Ranking"].map(s => (
                            <span key={s} className="step-chip active">{s}</span>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Gallery ── */}
            {results.length > 0 && (
                <div className="gallery-section">
                    <div className="gallery-header">
                        <div>
                            <h2>✦ Generated Results</h2>
                            <p>Click to select · Double-click for fullscreen Before/After</p>
                        </div>
                        <span style={{fontSize:'0.8rem',color:'var(--text-muted)'}}>{results.length} candidates ranked</span>
                    </div>

                    <div className="gallery-grid">
                        {results.map(result => (
                            <div key={result.id}
                                className={`result-card ${selectedId === result.id ? 'selected' : ''}`}
                                onClick={() => setSelectedId(result.id)}
                                onDoubleClick={() => setLightboxResult(result)}>
                                <img src={result.image} alt={`Result ${result.id}`} />
                                {selectedId === result.id && <div className="selected-badge">✔ SELECTED</div>}
                                {result.id === 0 && selectedId !== result.id && <div className="best-badge">🏆 BEST</div>}
                                <div className="card-actions">
                                    <button className="card-action-btn" title="Copy to clipboard"
                                        onClick={e => { e.stopPropagation(); copyToClipboard(result); }}>📋</button>
                                    <button className="card-action-btn" title="Fullscreen"
                                        onClick={e => { e.stopPropagation(); setLightboxResult(result); }}>⛶</button>
                                </div>
                                <div className="result-info">
                                    <div className="score-row"><span>CLIP Match</span><span>{result.clip_score}</span></div>
                                    <div className="score-row"><span>LPIPS Dist</span><span>{result.lpips_score}</span></div>
                                    <div className="score-row final-score"><span>Final Score</span><span>{result.final_score}</span></div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Before/After for selected result */}
                    {selectedResult && (
                        <div className="ba-section">
                            <div className="ba-section-header">
                                <h3>◀▶ Before / After — drag the slider</h3>
                                <div className="ba-section-actions">
                                    <button className="btn-secondary" style={{padding:'0.4rem 0.9rem',fontSize:'0.82rem'}}
                                        onClick={() => copyToClipboard(selectedResult)}>📋 Copy</button>
                                    <button className="btn-download" style={{padding:'0.4rem 1rem',fontSize:'0.82rem'}} onClick={handleDownload}>
                                        💾 Download
                                    </button>
                                </div>
                            </div>
                            <BeforeAfter original={image} generated={selectedResult.image} />
                        </div>
                    )}
                </div>
            )}

            {/* ── Dataset Modal ── */}
            {modalOpen && (
                <div className="modal-overlay" onClick={() => setModalOpen(false)}>
                    <div className="modal-box" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <div>
                                <h2>📁 MagicBrush Dataset</h2>
                                <p>Search → click sample → paint mask → press Enter to generate</p>
                            </div>
                            <button className="modal-close" onClick={() => setModalOpen(false)}>✕</button>
                        </div>
                        <input className="modal-search" type="text" autoFocus
                            placeholder="Search… e.g. 'cat', 'balloon', 'window', 'sky', 'dog'"
                            value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
                        <div className="modal-workflow-hint">
                            <span>① Click sample</span><span className="arrow">→</span>
                            <span>② Paint mask on image</span><span className="arrow">→</span>
                            <span>③ Press Enter / Generate</span>
                        </div>
                        {loadingSamples && <div className="modal-loading"><div className="spinner-ring" /><span>Loading from HuggingFace…</span></div>}
                        <div className="modal-grid">
                            {filteredSamples.map((s, i) => (
                                <div key={i} className="modal-card" onClick={() => loadSample(s)}>
                                    <img src={s.src} alt={s.prompt} />
                                    <div className="modal-card-overlay"><span>⬇ Load</span></div>
                                    <div className="modal-card-prompt">{s.prompt}</div>
                                </div>
                            ))}
                            {!loadingSamples && filteredSamples.length === 0 && (
                                <div style={{gridColumn:'1/-1',textAlign:'center',color:'var(--text-muted)',padding:'2rem'}}>
                                    No samples match "{searchQuery}".
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
