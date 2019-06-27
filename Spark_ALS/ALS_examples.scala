/**
	* Author:Charles·Van
	* E-mail:williananjhon@hotmail.com
	* Date:2019-06-27 16:41
	* Project:Personal_Recommendation_Algorithm
	* FileName:ALS_examples.scala
	* Version 1.0
	*/


package ALS

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SQLContext

case class Movie(movieId: Int,title:String,genres:Seq[String])
case class User(userId:Int,gender:String,age:Int,occupation:Int,zip:String)

object ALS_examples {
	def main(args: Array[String]): Unit = {
		val sc:SparkContext
		val sqlContext = new SQLContext(sc)
		import sqlContext.implicits._
		//Ratings analyst
		val ratingText=sc.textFile("file:/root/data/ratings.dat")
		ratingText.first()
		val ratingRDD=ratingText.map(parseRating).cache()
		println("Total number of ratings: "+ratingRDD.count())
		println("Total number of movies rated: "+ratingRDD.map(_.product).distinct().count())
		println("Total number of users who rated movies: "+ratingRDD.map(_.user).distinct().count())

		//Create DataFrames
		val ratingDF=ratingRDD.toDF()
		val movieDF=sc.textFile("file:/root/data/movies.dat").map(parseMovie).toDF()
		val userDF=sc.textFile("file:/root/data/users.dat").map(parseUser).toDF()
		ratingDF.printSchema()
		movieDF.printSchema()
		userDF.printSchema()
		ratingDF.registerTempTable("ratings")
		movieDF.registerTempTable("movies")
		userDF.registerTempTable("users")

		val result=sqlContext.sql("""select title,rmax,rmin,ucnt
from
(select product, max(rating) as rmax, min(rating) as rmin, count(distinct user) as ucnt
from ratings
group by product) ratingsCNT
join movies on product=movieId
order by ucnt desc""")
		result.show()

		val mostActiveUser=sqlContext.sql("""select user, count(*) as cnt
from ratings group by user order by cnt desc limit 10""")
		mostActiveUser.show()
		val result=sqlContext.sql("""select distinct title, rating
from ratings join movies on movieId=product
where user=4169 and rating>4""")
		result.show()

		//ALS
		val splits=ratingRDD.randomSplit(Array(0.8,0.2), 0L)
		val trainingSet=splits(0).cache()
		val testSet=splits(1).cache()
		trainingSet.count()
		testSet.count()
		val model=(new ALS().setRank(20).setIterations(10).run(trainingSet))

		val recomForTopUser=model.recommendProducts(4169,5)
		val movieTitle=movieDF.map(array=>(array(0),array(1))).collectAsMap();
		val recomResult=recomForTopUser.map(rating=>(movieTitle(rating.product),rating.rating)).foreach(println)

		val testUserProduct=testSet.map{
			case Rating(user,product,rating) => (user,product)
		}
		val testUserProductPredict=model.predict(testUserProduct)
		testUserProductPredict.take(10).mkString("\n")

		val testSetPair=testSet.map{
			case Rating(user,product,rating) => ((user,product),rating)
		}
		val predictionsPair=testUserProductPredict.map{
			case Rating(user,product,rating) => ((user,product),rating)
		}

		val joinTestPredict=testSetPair.join(predictionsPair)
		val mae=joinTestPredict.map{
			case ((user,product),(ratingT,ratingP)) =>
				val err=ratingT-ratingP
				Math.abs(err)
		}.mean()
		//FP,ratingT<=1, ratingP>=4
		val fp=joinTestPredict.filter{
			case ((user,product),(ratingT,ratingP)) =>
				(ratingT <=1 & ratingP >=4)
		}
		fp.count()
		val ratingTP=joinTestPredict.map{
			case ((user,product),(ratingT,ratingP))=>
				(ratingP,ratingT)
		}
		val evalutor=new RegressionMetrics(ratingTP)
		evalutor.meanAbsoluteError
		evalutor.rootMeanSquaredError
	}
	// Define parse function
	def parseMovie(str: String):Movie = {
		val fields = str.split("::")
		assert(fields.size == 3)
		Movie(fields(0).toInt,fields(1).toString,Seq(fields(2)))
	}
	def parseUser(str: String) : User = {
		val fields=str.split("::")
		assert(fields.size==5)
		User(fields(0).toInt, fields(1).toString, fields(2).toInt, fields(3).toInt, fields(4).toString)
	}
	def parseRating(str: String): Rating = {
		val fields=str.split("::")
		assert(fields.size==4)
		Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
	}

}
