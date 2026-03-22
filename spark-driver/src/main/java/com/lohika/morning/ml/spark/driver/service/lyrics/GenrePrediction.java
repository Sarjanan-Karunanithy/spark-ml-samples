package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.LinkedHashMap;
import java.util.Map;

public class GenrePrediction {

    private String genre;
    private Map<String, Double> probabilities;

    // ✅ NEW: accepts full probability array from all 7 genres
    public GenrePrediction(String genre, double[] probabilityArray) {
        this.genre = genre;

        // Map each probability to its genre name in order
        // Order must match Genre.java numeric values (0=POP, 1=COUNTRY ... 6=HIPHOP)
        Genre[] genres = {
                Genre.POP,
                Genre.COUNTRY,
                Genre.BLUES,
                Genre.JAZZ,
                Genre.REGGAE,
                Genre.ROCK,
                Genre.HIPHOP,
                Genre.SOUL
        };

        this.probabilities = new LinkedHashMap<>();
        for (int i = 0; i < genres.length; i++) {
            if (i < probabilityArray.length) {
                // Round to 4 decimal places for clean output
                double rounded = Math.round(probabilityArray[i] * 10000.0) / 10000.0;
                this.probabilities.put(genres[i].getName(), rounded);
            }
        }
    }

    // ✅ KEPT: fallback constructor when no probability available
    public GenrePrediction(String genre) {
        this.genre = genre;
        this.probabilities = new LinkedHashMap<>();
    }

    public String getGenre() {
        return genre;
    }

    public Map<String, Double> getProbabilities() {
        return probabilities;
    }

    // ✅ NEW: helper to get individual genre probability by name
    public Double getProbabilityForGenre(String genreName) {
        return probabilities.getOrDefault(genreName, 0.0);
    }

    @Override
    public String toString() {
        return "GenrePrediction{" +
                "genre='" + genre + '\'' +
                ", probabilities=" + probabilities +
                '}';
    }
}