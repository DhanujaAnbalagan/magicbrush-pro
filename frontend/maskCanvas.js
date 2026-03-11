const { useRef, useEffect, useState, forwardRef, useImperativeHandle, useCallback } = React;

const MaskCanvas = forwardRef(({ imageUrl }, ref) => {
    const canvasRef      = useRef(null);
    const overlayRef     = useRef(null);        // cursor preview canvas
    const maskCanvasRef  = useRef(document.createElement('canvas'));
    const imgRef         = useRef(null);
    const undoStack      = useRef([]);           // undo history of mask blobs
    const isDrawingRef   = useRef(false);

    const [brushSize, setBrushSize]   = useState(32);
    const [mode, setMode]             = useState('draw');
    const [maskOpacity, setMaskOpacity] = useState(0.55);
    const [cursorPos, setCursorPos]   = useState(null);

    // ── Expose methods ────────────────────────────────────────────
    useImperativeHandle(ref, () => ({
        getMaskBase64: () => maskCanvasRef.current.toDataURL('image/png'),
        hasPaintedMask: () => {
            const mc  = maskCanvasRef.current;
            const data = mc.getContext('2d').getImageData(0, 0, mc.width, mc.height).data;
            for (let i = 3; i < data.length; i += 4) if (data[i] > 10) return true;
            return false;
        },
        clearMask: () => { clearAll(); },
        undo: () => { undoLast(); },
    }));

    // ── Load image ───────────────────────────────────────────────
    useEffect(() => {
        if (!imageUrl) return;
        const img = new Image();
        img.onload = () => {
            imgRef.current = img;
            const w = img.naturalWidth, h = img.naturalHeight;
            [canvasRef, overlayRef, maskCanvasRef].forEach(r => {
                r.current.width  = w;
                r.current.height = h;
            });
            maskCanvasRef.current.getContext('2d').clearRect(0, 0, w, h);
            undoStack.current = [];
            renderFrame();
        };
        img.src = imageUrl;
    }, [imageUrl]);

    useEffect(() => { renderFrame(); }, [maskOpacity]);

    // ── Render ────────────────────────────────────────────────────
    const renderFrame = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !imgRef.current) return;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgRef.current, 0, 0);
        ctx.globalAlpha = maskOpacity;
        ctx.drawImage(maskCanvasRef.current, 0, 0);
        ctx.globalAlpha = 1;
    }, [maskOpacity]);

    // ── Undo ──────────────────────────────────────────────────────
    const saveUndoState = () => {
        const mc = maskCanvasRef.current;
        mc.toBlob(blob => undoStack.current.push(blob));
    };

    const undoLast = () => {
        if (undoStack.current.length === 0) return;
        const blob = undoStack.current.pop();
        const url  = URL.createObjectURL(blob);
        const img  = new Image();
        img.onload = () => {
            const ctx = maskCanvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
            ctx.drawImage(img, 0, 0);
            URL.revokeObjectURL(url);
            renderFrame();
        };
        img.src = url;
    };

    // ── Clear ─────────────────────────────────────────────────────
    const clearAll = () => {
        saveUndoState();
        const ctx = maskCanvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
        renderFrame();
    };

    // ── Coordinate helper ─────────────────────────────────────────
    const getPos = (e) => {
        const canvas = canvasRef.current;
        const rect   = canvas.getBoundingClientRect();
        const scaleX = canvas.width  / rect.width;
        const scaleY = canvas.height / rect.height;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY,
                 cx: clientX - rect.left, cy: clientY - rect.top };
    };

    // ── Cursor preview ────────────────────────────────────────────
    const drawCursor = (cx, cy) => {
        const oc = overlayRef.current;
        if (!oc) return;
        const ctx = oc.getContext('2d');
        ctx.clearRect(0, 0, oc.width, oc.height);
        const scale = oc.width / oc.getBoundingClientRect().width;
        const r = (brushSize / 2);
        ctx.beginPath();
        ctx.arc(cx * scale, cy * scale, r, 0, Math.PI * 2);
        ctx.strokeStyle = mode === 'draw' ? 'rgba(99,102,241,0.9)' : 'rgba(255,80,80,0.9)';
        ctx.lineWidth   = 2;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
    };

    const clearCursor = () => {
        const oc = overlayRef.current;
        if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height);
    };

    // ── Drawing ───────────────────────────────────────────────────
    const startDrawing = (e) => {
        e.preventDefault();
        saveUndoState();
        isDrawingRef.current = true;
        paintAt(e);
    };

    const stopDrawing = () => {
        isDrawingRef.current = false;
        maskCanvasRef.current.getContext('2d').beginPath();
    };

    const paintAt = (e) => {
        const { x, y, cx, cy } = getPos(e);
        drawCursor(cx, cy);
        if (!isDrawingRef.current) return;

        const maskCtx = maskCanvasRef.current.getContext('2d');
        maskCtx.lineWidth = brushSize;
        maskCtx.lineCap   = 'round';
        maskCtx.lineJoin  = 'round';

        if (mode === 'draw') {
            maskCtx.globalCompositeOperation = 'source-over';
            maskCtx.strokeStyle = 'rgba(99,102,241,0.9)';
        } else {
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.strokeStyle = 'rgba(0,0,0,1)';
        }
        maskCtx.lineTo(x, y);
        maskCtx.stroke();
        maskCtx.beginPath();
        maskCtx.moveTo(x, y);
        renderFrame();
    };

    // ── Keyboard shortcut: Ctrl+Z undo ───────────────────────────
    useEffect(() => {
        const handler = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undoLast(); }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    if (!imageUrl) return null;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', width: '100%' }}>
            {/* Toolbar */}
            <div className="brush-toolbar">
                <button className={`tool-btn ${mode === 'draw'  ? 'active' : ''}`} onClick={() => setMode('draw')}>
                    🖌️ Paint
                </button>
                <button className={`tool-btn ${mode === 'erase' ? 'active' : ''}`} onClick={() => setMode('erase')}>
                    🧽 Erase
                </button>
                <button className="tool-btn" onClick={undoLast} title="Ctrl+Z">
                    ↩ Undo
                </button>
                <button className="tool-btn" onClick={clearAll}>
                    🗑 Clear
                </button>
                <div className="toolbar-slider-group">
                    <span className="brush-size-label">Size {brushSize}px</span>
                    <input type="range" min="5" max="120" step="1"
                        value={brushSize} onChange={e => setBrushSize(Number(e.target.value))}
                        style={{ width: '70px' }} />
                </div>
                <div className="toolbar-slider-group">
                    <span className="brush-size-label">Opacity</span>
                    <input type="range" min="0.1" max="1" step="0.05"
                        value={maskOpacity} onChange={e => setMaskOpacity(Number(e.target.value))}
                        style={{ width: '60px' }} />
                </div>
            </div>

            {/* Canvas stack */}
            <div className="canvas-container" style={{ position: 'relative' }}>
                {/* Main canvas */}
                <canvas ref={canvasRef} style={{ display: 'block', width: '100%', cursor: 'none' }}
                    onMouseDown={startDrawing} onMouseUp={stopDrawing}
                    onMouseLeave={() => { stopDrawing(); clearCursor(); }}
                    onMouseMove={paintAt}
                    onTouchStart={startDrawing} onTouchEnd={stopDrawing} onTouchMove={paintAt} />
                {/* Cursor overlay canvas */}
                <canvas ref={overlayRef}
                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%',
                             pointerEvents: 'none', display: 'block' }} />
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                💡 Paint the area to edit · <kbd>Ctrl+Z</kbd> undo · <kbd>B</kbd> brush · <kbd>E</kbd> erase
            </div>
        </div>
    );
});

window.MaskCanvas = MaskCanvas;
