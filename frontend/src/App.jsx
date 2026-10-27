import { useEffect, useRef, useState } from "react";
import { getHealth, predictDigit } from "./api.js";

const CANVAS_SIZE = 280;   // the drawing area in pixels
const SEND_SIZE = 28;      // we shrink the drawing to 28 x 28 before sending it
const PEN_WIDTH = 22;      // thick pen, so the digit survives shrinking

// Shrink the big canvas to SEND_SIZE x SEND_SIZE and return a 2D array of grey values 0-255.
function canvasToGrayArray(canvas) {
  const small = document.createElement("canvas");   // an "offscreen" canvas that is never shown
  small.width = SEND_SIZE;
  small.height = SEND_SIZE;
  const ctx = small.getContext("2d");
  ctx.drawImage(canvas, 0, 0, SEND_SIZE, SEND_SIZE); // the browser averages pixels while shrinking
  const { data } = ctx.getImageData(0, 0, SEND_SIZE, SEND_SIZE); // [r, g, b, a, r, g, b, a, ...]

  const rows = [];
  for (let y = 0; y < SEND_SIZE; y++) {
    const row = [];
    for (let x = 0; x < SEND_SIZE; x++) {
      const i = (y * SEND_SIZE + x) * 4;
      row.push(Math.round((data[i] + data[i + 1] + data[i + 2]) / 3)); // grey = average of red, green, blue
    }
    rows.push(row);
  }
  return rows;
}

function App() {
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);     // is the mouse button / finger down?
  const lastPointRef = useRef(null);
  const [hasInk, setHasInk] = useState(false);
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Paint the canvas black once, and ask the API which model it uses.
  useEffect(() => {
    clearCanvas();
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  function clearCanvas() {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasInk(false);
    setResult(null);
    setError("");
  }

  // Mouse and touch positions are given in screen pixels; convert them to canvas pixels.
  function getPoint(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE,
      y: ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE,
    };
  }

  function drawLine(from, to) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = PEN_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  // Pointer events work for mouse, pen and touch.
  function handlePointerDown(event) {
    event.preventDefault();
    canvasRef.current.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    const point = getPoint(event);
    lastPointRef.current = point;
    drawLine(point, point);            // a single click makes a dot
    setHasInk(true);
  }

  function handlePointerMove(event) {
    if (!drawingRef.current) return;
    const point = getPoint(event);
    drawLine(lastPointRef.current, point);
    lastPointRef.current = point;
  }

  function handlePointerUp() {
    drawingRef.current = false;
  }

  async function handlePredict() {
    if (!hasInk) {
      setError("Draw a digit first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const image = canvasToGrayArray(canvasRef.current);
      const data = await predictDigit(image);
      setResult(data);
    } catch (err) {
      // field errors (422) come back as { image: "message" }
      const fieldMessages = Object.values(err.fields || {});
      setError(fieldMessages.length > 0 ? fieldMessages[0] : err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header>
        <h1>Digit Recogniser</h1>
        <p className="muted">Draw one digit (0–9) in the black box, then press Predict.</p>
        {health && (
          <p className="muted small">
            Model: {health.model}
            {health.using_fallback ? " (fallback — the Keras CNN is not loaded)" : ""}
          </p>
        )}
      </header>

      {error && <div className="alert">{error}</div>}

      <div className="layout">
        <section className="card">
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="board"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          />
          <div className="buttons">
            <button type="button" className="secondary" onClick={clearCanvas} disabled={loading}>
              Clear
            </button>
            <button type="button" onClick={handlePredict} disabled={loading}>
              {loading ? "Predicting…" : "Predict"}
            </button>
          </div>
          <p className="muted small">Tip: draw big and in the middle, like on paper.</p>
        </section>

        {result && (
          <section className="card result">
            <p className="muted">Predicted digit</p>
            <p className="digit">{result.digit}</p>
            <p className="muted small">Confidence: {(result.confidence * 100).toFixed(1)}%</p>

            <h2>Probability of each digit</h2>
            <div className="bars">
              {result.probabilities.map((p, digit) => (
                <div key={digit} className="bar-row">
                  <span className="bar-label">{digit}</span>
                  <div className="bar-track">
                    <div
                      className={digit === result.digit ? "bar-fill best" : "bar-fill"}
                      style={{ width: `${(p * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="bar-value">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            {result.model_input_8x8 && (
              <>
                <h2>What the model saw (8 × 8)</h2>
                <div className="preview">
                  {result.model_input_8x8.flat().map((value, i) => {
                    const grey = Math.round((value / 16) * 255);
                    return <div key={i} style={{ background: `rgb(${grey}, ${grey}, ${grey})` }} />;
                  })}
                </div>
              </>
            )}
            <p className="muted small">Model: {result.model}</p>
          </section>
        )}
      </div>
    </main>
  );
}

export default App;
