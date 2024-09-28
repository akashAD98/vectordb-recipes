# 🎶 Music Recommendation System using LanceDB

This project implements a music recommendation system using audio feature extraction and vector similarity search. By utilizing **LanceDB**, **PANNs** for audio tagging, and **Librosa** for audio feature extraction, the system finds and recommends tracks with similar audio characteristics based on a query song.

## 📝 Project Overview

The **Music Recommendation System** allows users to:
- Upload their own audio files (in MP3 format) or choose from pre-defined sample tracks.
- Extract audio embeddings using a **pretrained audio tagging model** (PANNs).
- Perform a vector similarity search to find similar tracks based on extracted audio features.
- Play back both the query track and the recommended tracks.

## 🚀 Key Features

- **Audio Embedding Extraction**: Extracts audio embeddings using **PANNs (Pretrained Audio Neural Networks)**.
- **Vector Similarity Search**: Utilizes **LanceDB** for fast vector similarity searches to recommend songs based on their audio embeddings.
- **Interactive UI**: Built using **Streamlit**, offering an intuitive interface to upload or select audio files and view recommendations.
- **Supports Various Genres**: Works with a dataset containing songs from multiple genres.

## 📦 Installation

To set up the project, follow these steps:

1. **Clone the Repository**:

2. **Install Required Dependencies**:
   Use `pip` to install all required libraries:
   ```bash
   pip install pandas datasets panns-inference numpy librosa ipython streamlit lancedb soundfile 
   ```

3. **Set up the Pretrained Model**:
   The pretrained PANNs model is used to extract audio embeddings. You can either:
   - Download the checkpoint automatically (handled by the app).
   - Or manually place the `Cnn14_mAP=0.431.pth` checkpoint in the `panns_data` folder.

## 🔧 Usage

1. **Run the Streamlit App**:
   Start the Streamlit app with the following command:
   ```bash
   streamlit run app_music.py
   ```

2. **Using the App**:
   - Upload a music file (MP3 format only) or choose one of the sample tracks.
   - The app will load the selected audio, extract features, and find similar tracks from the database.
   - You can listen to both the query track and recommended tracks directly in the app.

## 🎵 Dataset

The dataset used in this project includes:
- **MP3 files**: 30-second clips of songs from various genres.
- **MFCCs**: Extracted **Mel-frequency Cepstral Coefficients (MFCCs)** for each song, stored as `.npy` files.
- **Spectrograms**: Mel-frequency spectrograms of each song, stored as `.npy` files.
- **Labels**: Metadata stored in `labels.json`, which includes genres, subgenres, and moods for each track.

### Folder Structure
```
data/
├── mfccs/
│   └── genre/
├── mp3/
│   └── genre/
├── spectrogram/
│   └── genre/
├── labels.json
├── subgenres.json
```

## 🧠 How It Works

1. **Audio Embedding Extraction**:
   - The uploaded or selected song is loaded using **Librosa**, and features are extracted using **PANNs**.
   
2. **Vector Search with LanceDB**:
   - Once the audio embedding is generated, it is passed to **LanceDB** to perform a vector similarity search. This returns the top tracks that have similar audio features.

3. **Recommendations**:
   - The results of the similarity search are displayed in the app, and the user can play the recommended songs.


## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any features, bug fixes, or improvements.

