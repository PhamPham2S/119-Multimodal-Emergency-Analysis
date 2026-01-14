# Frontend Inference Plan

## Goal

Build a frontend that runs the existing audio+text pipeline for inference using
pretrained heads stored at:

- `./models/fusion_linear.pt`
- `./models/urgency_head.pt`
- `./models/sentiment_head.pt`

## Inputs

1) Random Sampling
   - Pick a random `.wav` + `.json` pair from `./data/Sample` using filename stem.
   - Build text from `json.utterances[*].text` (join with spaces).
2) Manual Upload
   - Upload `.wav` and `.txt` (or provide text in a textarea).

## Inference Pipeline (Backend Module)

Create a dedicated inference module (separate from training), e.g.
`src/core/infer.py`, with:

- Preprocessing
  - Audio: load with `soundfile`, resample to 16k, mono, peak normalize.
  - Text: tokenize with KcELECTRA tokenizer (`max_length`).
- Model assembly
  - Load HuBERT and KcELECTRA base encoders.
  - Create FusionModel with the same hidden sizes and fusion_dim.
  - Load weights:
    - `fusion_linear.pt` into the fusion projection layer
    - `urgency_head.pt` into urgency head
    - `sentiment_head.pt` into sentiment head
- Output
  - Return `urgency` label + logits
  - Return `sentiment` label + logits

## Label Mapping

Persist label orders used at training time:

- `urgency_order`: fixed as `["하", "중", "상"]`
- `sentiment_order`: save to `./models/labels.json` during training
  and load during inference to keep indices stable.

## Frontend UI

1) Input Section
   - Button: `Random Sampling`
   - Upload: `.wav` file, `.txt` file
   - Display selected filenames and transcript preview
2) Pipeline Section
   - Status indicator (loading models, running inference)
3) Output Section
   - Predicted urgency + logits
   - Predicted sentiment + logits
   - Optional probabilities (softmax / sigmoid)

## API Surface (if using a backend service)

- `GET /api/samples/random`
  - Returns: wav file or URL, transcript text, metadata
- `POST /api/infer`
  - Body: wav + text
  - Returns: labels + logits

## Error Handling

- Missing pair in random sampling -> retry
- Invalid wav or empty text -> show validation message
- Model file missing -> show setup error

## Performance Notes

- Load models once at server startup
- Cache tokenizer
- Limit max audio length and text length

## Usage

```python
pip install -r requirements.txt
python -m uvicorn service.backend.app:app --reload
```

## Acceptance Checks

- Random sample runs end-to-end and returns labels
- Manual upload runs end-to-end and returns labels
- Logits and labels match the expected label order
