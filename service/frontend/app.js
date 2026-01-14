const { useMemo, useState } = React;
const { createRoot } = ReactDOM;
const h = React.createElement;

const formatArray = (values) => values.map((value) => value.toFixed(4)).join(", ");

function App() {
  const [audioUrl, setAudioUrl] = useState("");
  const [samplePath, setSamplePath] = useState("");
  const [text, setText] = useState("");
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [audioFile, setAudioFile] = useState(null);

  const hasInput = useMemo(() => Boolean(samplePath || audioFile), [samplePath, audioFile]);

  const resetResult = () => {
    setResult(null);
    setError("");
  };

  const loadRandom = async () => {
    setStatus("Loading random sample...");
    resetResult();
    setAudioFile(null);
    try {
      const response = await fetch("/api/samples/random");
      if (!response.ok) {
        throw new Error("Failed to load sample");
      }
      const data = await response.json();
      setAudioUrl(data.audio_url || "");
      setSamplePath(data.audio_path || "");
      setText(data.text || "");
      setStatus("Sample loaded");
    } catch (err) {
      setStatus("Idle");
      setError(err.message);
    }
  };

  const handleAudioUpload = (event) => {
    resetResult();
    const file = event.target.files?.[0] || null;
    setAudioFile(file);
    setSamplePath("");
    if (file) {
      setAudioUrl(URL.createObjectURL(file));
      setStatus("Audio file ready");
    }
  };

  const handleTextUpload = (event) => {
    resetResult();
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setText(String(reader.result || ""));
    };
    reader.readAsText(file, "utf-8");
  };

  const runInference = async () => {
    if (!hasInput) {
      setError("Select an audio sample first.");
      return;
    }
    if (!text.trim()) {
      setError("Text is required for inference.");
      return;
    }
    resetResult();
    setStatus("Running inference...");

    try {
      let response;
      if (audioFile) {
        const formData = new FormData();
        formData.append("audio_file", audioFile);
        formData.append("text", text);
        response = await fetch("/api/infer-upload", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("/api/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_path: samplePath, text }),
        });
      }

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Inference failed");
      }
      const data = await response.json();
      setResult(data);
      setStatus("Inference complete");
    } catch (err) {
      setStatus("Idle");
      setError(err.message);
    }
  };

  const renderError = () => (error ? h("div", { className: "status" }, `Error: ${error}`) : null);
  const renderAudio = () =>
    audioUrl ? h("audio", { className: "audio-player", controls: true, src: audioUrl }) : null;

  const renderGold = () => {
    if (!result || !result.gold) {
      return null;
    }
    const urgencyMatch = result.urgency.label === result.gold.urgency;
    const sentimentMatch = result.sentiment.label === result.gold.sentiment;
    return h(
      "div",
      { className: "result-grid" },
      h(
        "div",
        null,
        h("div", { className: "chip" }, "Gold Urgency"),
        h(
          "div",
          { className: "section" },
          h("h4", null, result.gold.urgency),
          h(
            "span",
            { className: `badge ${urgencyMatch ? "match" : "mismatch"}` },
            urgencyMatch ? "match" : "mismatch"
          )
        )
      ),
      h(
        "div",
        null,
        h("div", { className: "chip" }, "Gold Sentiment"),
        h(
          "div",
          { className: "section" },
          h("h4", null, result.gold.sentiment),
          h(
            "span",
            { className: `badge ${sentimentMatch ? "match" : "mismatch"}` },
            sentimentMatch ? "match" : "mismatch"
          )
        )
      )
    );
  };

  const renderOutput = () => {
    if (!result) {
      return h("div", { className: "status" }, "Awaiting inference.");
    }
    return h(
      "div",
      { className: "result-grid" },
      h(
        "div",
        null,
        h("div", { className: "chip" }, "Urgency"),
        h("h4", null, result.urgency.label),
        h("div", { className: "logits" }, `logits: ${formatArray(result.urgency.logits)}`),
        h("div", { className: "logits" }, `probs: ${formatArray(result.urgency.probs)}`)
      ),
      h(
        "div",
        null,
        h("div", { className: "chip" }, "Sentiment"),
        h("h4", null, result.sentiment.label),
        h("div", { className: "logits" }, `logits: ${formatArray(result.sentiment.logits)}`),
        h("div", { className: "logits" }, `probs: ${formatArray(result.sentiment.probs)}`)
      )
    );
  };

  return h(
    "div",
    { className: "app" },
    h(
      "header",
      { className: "header" },
      h("div", { className: "title" }, "Emergency Fusion Inference"),
      h(
        "div",
        { className: "subtitle" },
        "Combine HuBERT audio embeddings with KcELECTRA text context to estimate urgency and sentiment in one pass."
      )
    ),
    h(
      "div",
      { className: "grid" },
      h(
        "section",
        { className: "card section" },
        h("h3", null, "Input"),
        h("button", { onClick: loadRandom }, "Random Sampling"),
        h(
          "div",
          null,
          h("label", null, "Upload WAV"),
          h("input", { type: "file", accept: "audio/wav", onChange: handleAudioUpload })
        ),
        h(
          "div",
          null,
          h("label", null, "Upload TXT"),
          h("input", { type: "file", accept: "text/plain", onChange: handleTextUpload })
        ),
        h(
          "div",
          null,
          h("label", null, "Transcript"),
          h("textarea", { value: text, onChange: (event) => setText(event.target.value) })
        )
      ),
      h(
        "section",
        { className: "card section" },
        h("h3", null, "Pipeline"),
        h("div", { className: "status" }, status),
        h(
          "button",
          { onClick: runInference, disabled: !hasInput },
          "Run Inference"
        ),
        renderError(),
        renderAudio()
      ),
      h(
        "section",
        { className: "card section" },
        h("h3", null, "Output"),
        renderOutput(),
        renderGold()
      )
    ),
    h("div", { className: "footer" }, "Powered by HuBERT + KcELECTRA fusion heads.")
  );
}

const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot
  ? createRoot(rootElement)
  : { render: (element) => ReactDOM.render(element, rootElement) };
root.render(h(App));
