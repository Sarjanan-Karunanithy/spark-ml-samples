package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;// ✅ NEW
import org.apache.spark.ml.feature.HashingTF;      // ✅ NEW
import org.apache.spark.ml.feature.IDF;            // ✅ NEW
import org.apache.spark.ml.feature.IDFModel;       // ✅ NEW
// ✅ Add import
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;

import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.stereotype.Component;

@Component("LogisticRegressionPipeline")
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

    // ✅ stores test accuracy after classify() runs
    private double testAccuracy = 0.0;

    public CrossValidatorModel classify() {
        Dataset<Row> sentences = readLyrics();

        // 80/20 split
        Dataset<Row>[] splits = sentences.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData     = splits[1];
        System.out.println("Training rows = " + trainingData.count());
        System.out.println("Testing  rows = " + testData.count());

        Cleanser cleanser       = new Cleanser();
        Numerator numerator     = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        Exploder exploder = new Exploder();
        Stemmer stemmer   = new Stemmer();
        Uniter uniter     = new Uniter();
        Verser verser     = new Verser();

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol(VERSE.getName())
                .setOutputCol("features")
                .setMinCount(0);

//        // HashingTF converts words array → numeric frequency vector
//        // numFeatures = vocabulary size (higher = more accurate, more memory)
//        HashingTF hashingTF = new HashingTF()
//                .setInputCol(VERSE.getName())
//                .setOutputCol("rawFeatures");
////                .setNumFeatures(10000); FILTERED_WORDS.getName())
//
//        // IDF weights rare words higher than common words
//        IDF idf = new IDF()
//                .setInputCol("rawFeatures")
//                .setOutputCol("features");

        LogisticRegression logisticRegression = new LogisticRegression()
                .setFamily("multinomial");

//        NaiveBayes naiveBayes = new NaiveBayes()
//                .setModelType("multinomial")
//                .setSmoothing(1.0);

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        cleanser,
                        numerator,
                        tokenizer,
                        stopWordsRemover,
                        exploder,
                        stemmer,
                        uniter,
                        verser,
                        word2Vec,
                        logisticRegression});
//        Pipeline pipeline = new Pipeline().setStages(
//                new PipelineStage[]{
//                        cleanser,
//                        numerator,
//                        tokenizer,
//                        stopWordsRemover,
//                        exploder,
//                        stemmer,
//                        uniter,
//                        verser,
//                        hashingTF,
//                        idf,
//                        naiveBayes});//logisticRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{8, 16, 32})//
                .addGrid(word2Vec.vectorSize(), new int[]{200, 300})
                .addGrid(logisticRegression.regParam(), new double[]{0.01D})
                .addGrid(logisticRegression.maxIter(), new int[]{100, 200})
                .build();
//        // ✅ BETTER PARAM GRID for TF-IDF
//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid(hashingTF.numFeatures(), new int[]{10000})
//                .addGrid(logisticRegression.regParam(), new double[]{0.01D})
//                .addGrid(logisticRegression.maxIter(), new int[]{100, 200})
//                .build();

//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16, 32})
//                .addGrid(hashingTF.numFeatures(), new int[]{4096, 8192})
//                .addGrid(idf.minDocFreq(), new int[]{0, 1, 2})
//                .addGrid(naiveBayes.smoothing(), new double[]{0.5, 1.0})
//                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setLabelCol(LABEL.getName())
                        .setPredictionCol("prediction")
                        .setMetricName("accuracy"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5);

        // train on 80%
        CrossValidatorModel model = crossValidator.fit(trainingData);

        // evaluate on 20%
        Dataset<Row> predictions = ((PipelineModel) model.bestModel()).transform(testData);
        this.testAccuracy = new MulticlassClassificationEvaluator()
                .setLabelCol(LABEL.getName())
                .setPredictionCol("prediction")
                .setMetricName("accuracy")
                .evaluate(predictions);

        System.out.println("\n================================================");
        System.out.println("Test Accuracy = " + String.format("%.2f%%", testAccuracy * 100));
        System.out.println("================================================\n");

        saveModel(model, getModelDirectory());

        return model;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse",  ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        modelStatistics.put("Vector size",         ((Word2VecModel) stages[8]).getVectorSize());
        modelStatistics.put("Reg parameter",       ((LogisticRegressionModel) stages[9]).getRegParam());
        modelStatistics.put("Max iterations",      ((LogisticRegressionModel) stages[9]).getMaxIter());

//        modelStatistics.put("Num features (TF-IDF)",
//                ((HashingTF) stages[3]).getNumFeatures());
//        modelStatistics.put("Reg parameter",
//                ((LogisticRegressionModel) stages[5]).getRegParam());
//        modelStatistics.put("Max iterations",
//                ((LogisticRegressionModel) stages[5]).getMaxIter());
//        modelStatistics.put("Num features",
//                ((HashingTF) stages[8]).getNumFeatures());
//        modelStatistics.put("Smoothing",
//                ((NaiveBayesModel) stages[10]).getSmoothing());
        modelStatistics.put("Test accuracy (20%)",
                String.format("%.2f%%", testAccuracy * 100));

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/logistic-regression/";
    }
}