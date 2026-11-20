# ML pipeline and evaluation

The optional classifier is Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593`, run on CPU unless the Transformers runtime selects otherwise. The audio is converted to mono and resampled to 16 kHz. Ten-second chunks are sent to the audio-classification pipeline; the top three labels and scores are returned per chunk. These are chunk-level classifications, not frame-level events, speech recognition, or speaker identification.

Weights are loaded lazily once per process. Download or initialization errors disable classification for that analysis while DSP results remain available. Scores are model confidence scores and are not calibrated guarantees.

No accuracy, precision, recall, F1, confusion matrix, per-class results, or inference benchmarks are claimed. A credible evaluation must use a fixed labeled test split (for example ESC-50 or UrbanSound8K), map dataset labels to the model label vocabulary, document aggregation from chunk predictions to clips, and report macro metrics and per-class confusion results.
