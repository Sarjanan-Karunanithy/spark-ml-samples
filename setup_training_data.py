"""
setup_training_data.py

Converts tcc_ceds_music.csv into the folder structure
that tmatyashovsky/spark-ml-samples expects:

training-set/verse1/
    pop/
        pop_0.txt
        pop_1.txt ...
    country/
        country_0.txt ...
    blues/ ...
    rock/ ...
    jazz/ ...
    reggae/ ...
    hip hop/
        hiphop_0.txt ...

HOW TO RUN:
    python setup_training_data.py

BEFORE RUNNING:
    - Put this script in the root of your cloned repo
    - Put tcc_ceds_music.csv in the same folder as this script
    - Run: python setup_training_data.py
"""

import csv
import os

# ─────────────────────────────────────────────
# CONFIG — adjust these paths if needed
# ─────────────────────────────────────────────
CSV_FILE = "Merged_dataset.csv" #tcc_ceds_music.csv"                        # input CSV
OUTPUT_DIR = os.path.join("training-set", "verse1")    # output folder
SONGS_PER_FILE = 50   # group N songs per .txt file (keeps file count manageable)
# ─────────────────────────────────────────────

# Safe folder names (no spaces/special chars)
GENRE_FOLDER_MAP = {
    "pop":     "pop",
    "country": "country",
    "blues":   "blues",
    "rock":    "rock",
    "jazz":    "jazz",
    "reggae":  "reggae",
    "hip hop": "hiphop",
    "soul": "soul",
}

def setup():
    # 1. Create output genre folders
    for folder in GENRE_FOLDER_MAP.values():
        path = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(path, exist_ok=True)
    
    # Also create model folder
    os.makedirs(os.path.join(OUTPUT_DIR, "model"), exist_ok=True)

    print(f"✅ Created folder structure under: {OUTPUT_DIR}")

    # 2. Read CSV and group lyrics by genre
    genre_lyrics = {g: [] for g in GENRE_FOLDER_MAP.values()}

    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre_raw = row["genre"].strip().lower()
            lyrics    = row["lyrics"].strip()
            
            if genre_raw in GENRE_FOLDER_MAP and lyrics:
                folder = GENRE_FOLDER_MAP[genre_raw]
                genre_lyrics[folder].append(lyrics)

    # 3. Write .txt files — one file per SONGS_PER_FILE songs
    total_files = 0
    for genre, lyrics_list in genre_lyrics.items():
        file_index = 0
        for i in range(0, len(lyrics_list), SONGS_PER_FILE):
            chunk = lyrics_list[i : i + SONGS_PER_FILE]
            filename = f"{genre}_{file_index}.txt"
            filepath = os.path.join(OUTPUT_DIR, genre, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                # Write each song lyric as one line (how the repo reads it)
                for lyric in chunk:
                    f.write(lyric + "\n")
            
            file_index += 1
            total_files += 1

        print(f"  📁 {genre:10s} → {len(lyrics_list):5d} songs → {file_index} files")

    print(f"\n✅ Done! {total_files} total .txt files created.")
    print(f"\n📌 Now update your papplication.properties:")
    abs_path = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    model_path = os.path.abspath(os.path.join(OUTPUT_DIR, "model")).replace("\\", "/")
    print(f"\n  verse1.training.set.directory.path={abs_path}/")
    print(f"  verse1.model.directory.path={model_path}")

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"❌ ERROR: '{CSV_FILE}' not found!")
        print(f"   Place tcc_ceds_music.csv in the same folder as this script.")
    else:
        setup()
