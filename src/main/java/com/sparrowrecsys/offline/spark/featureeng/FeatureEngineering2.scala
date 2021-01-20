package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{MinMaxScaler, OneHotEncoderEstimator, QuantileDiscretizer, StringIndexer, StringIndexerModel}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object FeatureEngineering2 {

  def oneHotEncoderExample(samples: DataFrame): Unit = {
    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      .setDropLast(false)

    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)
  }

  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }

  def multiHotEncoderExample(samples: DataFrame): Unit = {
    val samplesWithGenre = samples.select(col("movieId"), col("title"), explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
    println("samplesWithGenre:")
    samplesWithGenre.printSchema()
    samplesWithGenre.show(10)

    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")

    val stringIndexerModel: StringIndexerModel = genreIndexer.fit(samplesWithGenre)

    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex")
        .cast(sql.types.IntegerType))

    println("genreIndexSamples:")
    genreIndexSamples.printSchema()
    genreIndexSamples.show(10)

    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    val processedSamples = genreIndexSamples
      .groupBy(col("movieId"))
      .agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize))

    println("processedSamples:")
    processedSamples.printSchema()
    processedSamples.show(10)

    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10)
  }

  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

  def ratingFeatures(samples: DataFrame): Unit = {
    print("samples:")
    samples.printSchema()
    samples.show(10)

    //calculate average movie rating score and rating count
    val movieFeatures = samples.groupBy("movieId")
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))
      .withColumn("avgRatingVec", double2vec(col("avgRating")))

    movieFeatures.show(10)

    //bucketing
    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    //Normaliztion
    val ratingScaler = new MinMaxScaler()
      .setInputCol("avgRatingVec")
      .setOutputCol("scaleAvgRating")

    val pipelinStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    val featurePipeline = new Pipeline().setStages(pipelinStage)

    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
    conf.setMaster("local")
      .setAppName("featureEngineering2")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResouresPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val moviesSamples = spark.read.format("csv").option("header", "true").load(movieResouresPath.getPath)
    println("Raw Movie Samples:")
    moviesSamples.printSchema()
    moviesSamples.show(10)

    println("OneHotEncoder Example:")
    oneHotEncoderExample(moviesSamples)

    println("MultiHotEncoder Examples:")
    multiHotEncoderExample(moviesSamples)

    println("Numerical features Example:")
    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath.getPath)
    ratingFeatures(ratingSamples)
  }

}
