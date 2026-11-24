# Movie Sentiment Analysis with SimpleRNN

A Streamlit application that predicts whether a movie review is positive or
negative using a recurrent neural network trained on the IMDb reviews dataset.

## Features

- Interactive Streamlit interface for entering a review.
- Bundled trained model (`simple_rnn_model.h5`), so the app does not need to be
  trained before it can run.
- Jupyter notebooks documenting the IMDb model workflow, word embeddings, and
  prediction preprocessing.

## Model

The training notebook uses the IMDb dataset provided by Keras, limits the
vocabulary to 10,000 words, and pads reviews to 500 tokens. The model consists
of an embedding layer, a 128-unit `SimpleRNN` layer, and a sigmoid output layer.
Training uses Adam, binary cross-entropy, and early stopping.

The model returns a score between 0 and 1. The app labels scores above 0.5 as
positive and other scores as negative. This is a learning project; predictions
may be inaccurate, especially for text unlike the IMDb training data.

## Project files

| File | Description |
| --- | --- |
| `main.py` | Streamlit application and review prediction code |
| `simple_rnn_model.h5` | Pre-trained Keras model used by the app |
| `srrn.ipynb` | IMDb dataset preparation, SimpleRNN training, and model saving |
| `prediction.ipynb` | Notebook examples for review preprocessing and prediction |
| `embbeding.ipynb` | Word embedding demonstration |
| `requirements.txt` | Python runtime dependencies |

## Requirements

- Python 3.10–3.12
- pip

## Run locally

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/pratyushkothari5/movie-sentiment-analysis-using-SimpleRNN.git
   cd movie-sentiment-analysis-using-SimpleRNN
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies and launch the app:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run main.py
   ```

Streamlit prints a local URL in the terminal. Open it in a browser to submit a
review.

## Retraining

Open `srrn.ipynb` and run its cells to load and preprocess the IMDb dataset,
train the network, and save a replacement `simple_rnn_model.h5`. The first
dataset load may require an internet connection to download the Keras IMDb
dataset.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
