/**
	* Author:Charles·Van
	* E-mail:williananjhon@hotmail.com
	* Date:2019-06-27 20:50
	* Project:Personal_Recommendation_Algorithm
	* FileName:MovieLensALS.scala
	* Version 1.0
	*/


package ALS


import java.io.File

import org.apache.log4j.{Level,Logger}
import org.apache.spark.{SparkContext,SparkConf}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel,Rating,ALS}
import org.apache.spark.rdd.RDD
import scala.util.Random


object MovieLensALS {
	// define a ratting elicitation function
	def elicitateRating(movies:Seq[(Int,String)]) = {
		val prompt = "Please rate the following movie(1-5(best) or 0 if not seen:)"
		println(prompt)
		val ratings = movies.flatMap{x =>
			val rating: Option[Rating] = None
			val valid = false
			while (!valid){
				println(x._2 + " :")
				try{
					val r = Console.readInt()
					if (r > 5 || r < 0){
						println(prompt)
					}
					else{
						valid = true
						if (r > 0){
							rating = Some(Rating(0,x._1,r))
						}
					}
				}
				catch{
					case e:Exception => println(prompt)
				}
			}
			rating match{
				case Some(r) => Iterator(r)
				case None => Iterator.empty
			}
		}
		if (ratings.isEmpty){
			error("No ratings provided!")
		}
		else {
			ratings
		}
	}
	// Define a RMSE computation function
	def computeRMSE(model:MatrixFactorizationModel,data:RDD[Rating]) = {
		val prediction = model.predict(data.map(x => (x.user,x.product)))
		val predDataJoined = prediction.map(x =>
			((x.user,x.product),x.rating)).join(data.map(x =>
			((x.user,x.product),x.rating))).values
		new RegressionMetrics(predDataJoined).rootMeanSquaredError
	}

	// Main
	def main(args: Array[String]): Unit = {
		// setup env
		Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

		if (args.length != 1){
			print("Usage: movieLensHomeDir")
			sys.exit(1)
		}

		val conf = new SparkConf().setAppName("MovieLensALS")
			.set("spark.executor.memory","500m")
		val sc = new SparkContext(conf)

		// Load ratings data and know your data
		val movieLensHomeDir = args(0)
		val ratings = sc.textFile(new File(movieLensHomeDir,
			"ratings.dat").toString).map{line =>
			val fields = line.split("::")
			// timestamp,user,product,rating
			(fields(3).toLong%10,Rating(fields(0).toInt,fields(1).toInt,
				fields(2).toDouble))
		}
		val movies = sc.textFile(new File(movieLensHomeDir,
		"movies.dat").toString).map{line =>
			val fields = line.split("::")
			// movieId.movieName
			(fields(0).toInt,fields(1))
		}.collectAsMap()

		val numRatings = ratings.count()
		val numUser = ratings.map(_._2.user).distinct().count()
		val numMovie = ratings.map(_._2.product).distinct().count()

		println("Got " + numRatings + " ratings from " + numUser + " users on " + numMovie + " movies")

		// Elicitate personal rating
		val topMoives = ratings.map(_._2.product).countByValue().toSeq.sortBy(-_._2).take(50).map(_._1)
		val random = new Random(0)
		val selectMovies = topMoives.filter(x => random.nextDouble() < 0.2).map(x => (x,movies(x)))

		val myRatings = elicitateRating(selectMovies)
		val myRatingRDD = sc.parallelize(myRatings,1)

		// Split data into train(60%),validation(20%) and test(20%)
		val numPartitions = 10
		val trainSet = ratings.filter(_._1 < 6).map(_._2).union(myRatingRDD).repartition(numPartitions).persist()
		val validationSet = ratings.filter(x => x._1 > 6 && x._1 < 8).map(_._2).persist()
		val testSet = ratings.filter(_._1 > 8).map(_._2).persist()

		val numTrain = trainSet.count()
		val numValidation = validationSet.count()
		val numTest = testSet.count()

		println("Training data: " + numTrain + " Validation data: " + numValidation + " Test data: " + numTest)

		// Train model and optimize model with validation set
		val numRanks = List(8,12)
		val numIters = List(10,20)
		val numLambdas = List(0.1,10.0)
		var bestRMSE = Double.MaxValue
		var bestModel: Option[MatrixFactorizationModel] = None
		var bestRanks = -1
		var bestIters = 0
		var bestLambdas = -1.0
		for (rank <- numRanks;iter <- numIters;lambda <- numLambdas){
			val model = ALS.train(trainSet,rank,iter,lambda)
			val validationRMSE = computeRMSE(model,validationSet)
			println("RMSE(validation) = " + validationRMSE + " with ranks = " + rank + ", iter = " + iter + ", Lambda = " + lambda)

			if (validationRMSE < bestRMSE){
				bestModel = Some(model)
				bestRMSE = validationRMSE
				bestIters = iter
				bestLambdas = lambda
				bestRanks = rank
			}
		}

		// Evaluate model on test set
		val testRMSE = computeRMSE(bestModel.get,testSet)
		println("The best model was trained with rank="+bestRanks+", Iter="+bestIters+", Lambda="+bestLambdas+
			" and compute RMSE on test is "+testRMSE)

		// Create a baseline and compare it with best model
		val meanRating = trainSet.union(validationSet).map(_.rating).mean()
		val bestlineRMSE = new RegressionMetrics(testSet.map(x =>
			(x.rating,meanRating))).rootMeanSquaredError
		val improvement = (bestlineRMSE - testRMSE) / bestlineRMSE * 100
		println("The best model improves the baseline by "+"%1.2f".format(improvement)+"%.")

		// Make a personal recommendation
		val moviesId = myRatings.map(_.product)
		val candidates = sc.parallelize(movies.keys.filter(!moviesId.contains(_)).toSeq)
		val recommendations = bestModel.get
			.predict(candidates.map(x=>(0, x)))
			.sortBy(-_.rating)
			.take(50)

		val i = 0
		println("Movies recommended for you:")
		recommendations.foreach{ line=>
			println("%2d".format(i)+" :"+movies(line.product))
			i += 1
		}
		sc.stop()
	}
}
