import joblib
import gradio as gr
import librosa
import numpy as np

# Define the genre mapping
genre_mapping = {0: 'Classical', 1: 'Jazz', 2: 'Pop', 3: 'Rock', 4: 'Hip-Hop', 5: 'Electronic', 6: 'Country', 7: 'Blues', 8: 'Reggae', 9: 'Metal'}

def extract_features(y, sr):
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    
    expected_length = 58
    if len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)), 'constant')
    elif len(features) > expected_length:
        features = features[:expected_length]
    
    print(f"Adjusted features shape: {features.shape}")  # This should print (58,)
    
    return features

def predict_genre(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)
    features = extract_features(y, sr)
    
    model = joblib.load('model.pkl')
    
    print(f"Shape of features before prediction: {features.shape}")  # This should print (58,)
    
    # Reshape features to match model input
    features = np.expand_dims(features, axis=0)
    print(f"Shape of features after reshaping: {features.shape}")  # This should print (1, 58)
    
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_genre = genre_mapping[predicted_index]
    
    return predicted_genre

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_genre,  # The function to wrap
    inputs=gr.Audio( type="filepath"),  # Input type
    outputs="text",  # Output type
    title="Music Genre Classification",  # Title of the interface
    description="Upload an audio file to classify its music genre."  # Description
)

# Launch the Gradio interface
interface.launch(share=True,debug=True)
