# Music Genre Classifier

This project classifies the genre of a given audio file using a pre-trained machine learning model. The project uses FastAPI for the web interface and Librosa for audio processing.

## Project Structure

```
Music-Genre-Classifier/
├── main.py
├── main.ipynb
├── rf.pkl
├── scaler.pkl
├── requirements.txt
├── static/
│   └── style.css
└── README.md
```

- `main.py`: The FastAPI application script.
- `main.ipynb`: The Jupyter Notebook used for training the machine learning model.
- `rf.pkl`: The pre-trained RandomForest model.
- `scaler.pkl`: The scaler used to preprocess the data.
- `requirements.txt`: The list of dependencies.
- `static/`: Directory containing static files (e.g., CSS for the web interface).
- `README.md`: This readme file.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Librosa
- Joblib
- Scikit-learn
- Jupyter Notebook (for training the model)


## Training the Model

The model can be trained using the `main.ipynb` Jupyter Notebook. This notebook contains the steps to load the dataset, preprocess the audio files, extract features, train the RandomForest model, and save the model and scaler.

1. Launch Jupyter Notebook:

```sh
jupyter notebook
```

2. Open `main.ipynb` and run all the cells to train the model.

## Running the Application

1. Ensure the `rf.pkl` and `scaler.pkl` files are in the project directory. These files are generated after running `main.ipynb`.

2. Start the FastAPI server:

```sh
uvicorn main:app --reload
```

3. Open your browser and navigate to `http://127.0.0.1:8000`. You should see a web interface where you can upload an audio file to classify its genre.

## File Upload and Classification

- Upload an audio file (e.g., .wav format) using the web interface.
- The server processes the audio file, extracts relevant features using Librosa, scales the features, and predicts the genre using the pre-trained model.
- The predicted genre is displayed on the web interface.

## CSS and Spinner

The web interface includes a spinner that shows while the file is being processed. The CSS for the spinner is located in `static/style.css`.


## Acknowledgements

- [Librosa](https://librosa.org/doc/latest/index.html) for audio processing.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning tools.
```
