import librosa
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Load the model and scaler
model = joblib.load('rf.pkl')
scaler = joblib.load('scaler.pkl')

def getmetadata(filename):
    y, sr = librosa.load(filename)
    
    # Use keyword arguments for onset_strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    
    # Use harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    
    # Feature extraction
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y+0.01, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # Metadata dictionary
    metadata_dict = {
        'tempo': np.mean(tempo),
        'beats': np.mean(beat_frames),
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spec_centroid),
        'spectral_bandwidth': np.mean(spec_bw),
        'rolloff': np.mean(spec_rolloff),
        'zero_crossing_rates': np.mean(zero_crossing)
    }
    
    for i in range(1, 21):
        if i <= mfcc.shape[0]:  # Check if the index is within the bounds of mfcc
            metadata_dict[f'mfcc{i}'] = np.mean(mfcc[i-1])
    
    return metadata_dict

genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

def find_genre(file_path):
    mtdt = getmetadata(file_path)
    mtdt = np.array(list(mtdt.values()))
    mtdt.reshape(-1,1)
    mtdt_scaled = scaler.transform([mtdt])
    pred_genre = model.predict(mtdt_scaled) 
    return genre_mapping[pred_genre[0]]

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Music Genre Classifier</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Upload an audio file to classify its genre</h1>
                <form action="/uploadfile" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" required>
                    <input type="submit" value="Upload">
                </form>
                <div id="output"></div>
                <div class="spinner" id="spinner"></div>
            </div>
            <script>
                const form = document.querySelector('form');
                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const fileInput = document.querySelector('input[type="file"]');
                    const outputDiv = document.getElementById('output');
                    const spinner = document.getElementById('spinner');
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);

                    spinner.style.display = 'block';
                    outputDiv.textContent = '';
                    
                    const response = await fetch('/uploadfile', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    spinner.style.display = 'none';
                    outputDiv.innerHTML = `Predicted Genre: <strong>${result.genre}</strong>`;
                });
            </script>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        buffer.write(file.file.read())
    genre = find_genre("temp.wav")
    return {"genre": genre}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)