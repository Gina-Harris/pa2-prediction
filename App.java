package org.example.basicapp;
import org.apache.spark.*;
import org.apache.spark.sql.*;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class App 
{
    public static void main( String[] args )
    {
        SparkConf sparkConf = new SparkConf().setAppName("Training").setMaster("local").set("spark.driver.host", "localhost").set("headers","true");
        SparkSession spark = SparkSession
        .builder()
        .config(sparkConf)
        .getOrCreate();

        String testDataFilePath = "/data/TestValidationDataset.csv";
        Dataset<Row> testData = spark.read().option("delimiter", ";").option("header", "true").format("csv").load(testDataFilePath);

        testData = formatData(testData);

        PipelineModel plm1 = PipelineModel.load("path/model");

        Dataset<Row> predictionDF = plm1.transform(testData).cache();
        predictionDF.select("features", "quality", "prediction").show(5, false);

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionDF.select("prediction", "quality"));

        // Overall statistics
        System.out.println("Accuracy = " + metrics.accuracy());

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++) {
        System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
            metrics.labels()[i]));
        System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
            metrics.labels()[i]));
        System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
            metrics.labels()[i]));
        }

        //Weighted stats
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
        System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
        System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
        System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

        
        spark.stop();
    }

    public static Dataset<Row> formatData(Dataset<Row> df)
    {
        for (String col : df.columns()) 
        {
        df = df.withColumnRenamed(col, col.replace("\"",""));
        }

        for (String col : df.columns())
        {
        df = df.withColumn(col, df.col(col).cast("double"));
        }

        String[] cols = {"alcohol", "sulphates", "pH", "density", "free sulfur dioxide", "total sulfur dioxide", "chlorides", "residual sugar", "citric acid", "volatile acidity", "fixed acidity"};
        VectorAssembler va = new VectorAssembler().setInputCols(cols).setOutputCol("features");

        df = va.transform(df).select("quality","features");

        //df.show(5);

        return df;
    }
}
