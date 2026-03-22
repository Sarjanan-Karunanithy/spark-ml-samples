package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Override
    public GenrePrediction predict(final String unknownLyrics) {
        String lyrics[] = unknownLyrics.split("\\r?\\n");
        Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics),
                Encoders.STRING());

        Dataset<Row> unknownLyricsDataset = lyricsDataset
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                .withColumn(ID.getName(), functions.lit("unknown.txt"));

        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());
        getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
        Row predictionRow = predictionsDataset.first();

        System.out.println("\n------------------------------------------------");
        final Double prediction = predictionRow.getAs("prediction");
        System.out.println("Prediction: " + Double.toString(prediction));

        if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
            final DenseVector probability = predictionRow.getAs("probability");
            System.out.println("Probability: " + probability);
            System.out.println("------------------------------------------------\n");

            return new GenrePrediction(
                    getGenre(prediction).getName(),
                    probability.toArray()
            );
        }

        System.out.println("------------------------------------------------\n");
        return new GenrePrediction(getGenre(prediction).getName());
    }

    Dataset<Row> readLyrics() {
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

        Dataset<Row> input = readLyricsForGenre(lyricsTrainingSetDirectoryPath, genres[0]);
        for (int i = 1; i < genres.length; i++) {
            input = input.union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, genres[i]));
        }

        input = input.coalesce(sparkSession.sparkContext().defaultMinPartitions()).cache();
        input.count();

        return input;
    }

    // ✅ ADD this helper method anywhere in the class
    private String toSparkPath(String path) {
        String clean = path
                .replace("file:///", "")  // remove if already has it
                .replace("\\", "/");      // convert backslashes
        // add file:/// for Windows drive paths like C:/...
        if (clean.matches("^[A-Za-z]:.*")) {
            return "file:///" + clean;
        }
        return clean;
    }

    // ✅ FIXED: no wildcard — Spark reads entire folder directly
    private Dataset<Row> readLyricsForGenre(String inputDirectory, Genre genre) {
        //String genreFolder = Paths.get(inputDirectory, genre.name().toLowerCase()).toString();
        // ✅ FIXED: add file:/// prefix so Spark recognises it as a local path on Windows
        //String genreFolder = "file:///" +
        //        Paths.get(inputDirectory, genre.name().toLowerCase())
        //                .toString()
        //                .replace("\\", "/");  // ✅ also convert backslashes to forward slashes

        String genreFolder = toSparkPath(inputDirectory)
                + "/" + genre.name().toLowerCase();
        System.out.println("Reading from: " + genreFolder); // helpful debug line

        Dataset<String> rawLyrics = sparkSession.read().textFile(genreFolder);
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).notEqual(""));
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).contains(" "));

        Dataset<Row> lyrics = rawLyrics.withColumn(ID.getName(), functions.input_file_name());
        Dataset<Row> labeledLyrics = lyrics.withColumn(LABEL.getName(), functions.lit(genre.getValue()));

        System.out.println(genre.name() + " music sentences = " + lyrics.count());

        return labeledLyrics;
    }

    private Genre getGenre(Double value) {
        for (Genre genre : Genre.values()) {
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }
        return Genre.UNKNOWN;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best model metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        return modelStatistics;
    }

    void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    void saveModel(CrossValidatorModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(PipelineModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    public void setLyricsTrainingSetDirectoryPath(String lyricsTrainingSetDirectoryPath) {
        this.lyricsTrainingSetDirectoryPath = lyricsTrainingSetDirectoryPath;
    }

    public void setLyricsModelDirectoryPath(String lyricsModelDirectoryPath) {
        this.lyricsModelDirectoryPath = lyricsModelDirectoryPath;
    }

    protected abstract String getModelDirectory();

    String getLyricsModelDirectoryPath() {
        return lyricsModelDirectoryPath;
    }
}