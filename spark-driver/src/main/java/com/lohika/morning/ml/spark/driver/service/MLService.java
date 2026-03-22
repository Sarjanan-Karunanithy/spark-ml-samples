package com.lohika.morning.ml.spark.driver.service;

import java.io.IOException;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class MLService {

    @Autowired
    private SparkSession sparkSession;

    // ✅ ADD THIS — converts any Windows path to Spark-safe path
    private String toSparkPath(String path) {
        // Remove file:/// if already present (avoid double prefix)
        String clean = path
                .replace("file:///", "")
                .replace("\\", "/");
        // Add file:/// for Windows drive paths e.g. C:/... D:/...
        if (clean.matches("^[A-Za-z]:.*")) {
            return "file:///" + clean;
        }
        return clean;
    }

    public void trainLinearRegression(String fullSetParquetFilePath, int numIterations) {
        Tuple2<Dataset<Row>, Dataset<Row>> trainingAndTestDataSet =
                getTrainingAndTestDatasets(fullSetParquetFilePath);
        trainLinearRegression(
                trainingAndTestDataSet._1(),
                trainingAndTestDataSet._2(),
                numIterations);
    }

    public KMeansModel trainKMeans(String fullSetParquetFilePath, Integer clusters) {
        Dataset<Row> dataset = getTrainingDataset(fullSetParquetFilePath);
        KMeans kmeans = new KMeans().setK(clusters);
        return kmeans.fit(dataset);
    }

    private Tuple2<Dataset<Row>, Dataset<Row>> getTrainingAndTestDatasets(
            final String fullSetParquetFilePath) {
        Dataset<Row> fullSetDataFrame = sparkSession.read()
                .parquet(fullSetParquetFilePath);
        Dataset<Row>[] splitDataFrames =
                fullSetDataFrame.randomSplit(new double[]{0.7D, 0.3D});
        splitDataFrames[0].cache();
        splitDataFrames[0].count();
        return new Tuple2<>(splitDataFrames[0], splitDataFrames[1]);
    }

    private Dataset<Row> getTrainingDataset(final String fullSetParquetFilePath) {
        Dataset<Row> dataset = sparkSession.read().parquet(fullSetParquetFilePath);
        dataset.cache();
        dataset.count();
        return dataset;
    }

    private void trainLinearRegression(
            Dataset<Row> trainingSet,
            Dataset<Row> testSet,
            int numIterations) {
        LinearRegression linearRegression = new LinearRegression()
                .setMaxIter(numIterations)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        LinearRegressionModel linearRegressionModel =
                linearRegression.fit(trainingSet);

        System.out.println("Coefficients: " +
                linearRegressionModel.coefficients());

        LinearRegressionTrainingSummary trainingSummary =
                linearRegressionModel.summary();
        System.out.println("Total iterations: " +
                trainingSummary.totalIterations());
        System.out.println("Root mean squared error: " +
                trainingSummary.rootMeanSquaredError());

        Dataset<Row> predictions =
                linearRegressionModel.transform(testSet);
        Row predictedRow = predictions
                .select("features", "label", "prediction").first();
        System.out.println("(" + predictedRow.get(0) + ", " +
                predictedRow.get(1) + ", prediction=" + predictedRow.get(2));
    }

    // ✅ FIXED — saves model with Spark-safe path
    public <T extends MLWritable> void saveModel(
            T model, String modelOutputDirectory) {
        try {
            String path = toSparkPath(modelOutputDirectory);

            model.write().overwrite().save(path);

            System.out.println("\n------------------------------------------------");
            System.out.println("Saved model to " + path);
            System.out.println("------------------------------------------------\n");
        } catch (IOException e) {
            throw new RuntimeException(String.format(
                    "Exception occurred while saving the model. Details: %s",
                    e.getMessage()));
        }
    }

    // ✅ FIXED — loads model with Spark-safe path
    public CrossValidatorModel loadCrossValidationModel(String modelDirectory) {
        String path = toSparkPath(modelDirectory);

        System.out.println("\n------------------------------------------------");
        System.out.println("Loading model from: " + path);
        System.out.println("------------------------------------------------\n");

        CrossValidatorModel model = CrossValidatorModel.load(path);

        System.out.println("Loaded cross validation model successfully!");
        return model;
    }

    // ✅ FIXED — loads pipeline model with Spark-safe path
    public PipelineModel loadPipelineModel(String modelDirectory) {
        String path = toSparkPath(modelDirectory);

        System.out.println("\n------------------------------------------------");
        System.out.println("Loading pipeline model from: " + path);
        System.out.println("------------------------------------------------\n");

        PipelineModel model = PipelineModel.load(path);

        System.out.println("Loaded pipeline model successfully!");
        return model;
    }
}