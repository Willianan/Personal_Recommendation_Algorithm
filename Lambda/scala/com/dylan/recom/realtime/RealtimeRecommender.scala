package com.dylan.recom.realtime

import com.alibaba.fastjson.JSON
import com.dylan.recom.common.{Constants, RedisUtil}
import kafka.serializer.StringDecoder

import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._
import org.apache.spark.SparkConf

/**
 * Created by dylan
 */
object RealtimeRecommender {
  def main(args: Array[String]) {

    val Array(brokers, topics) = Array(Constants.KAFKA_ADDR, Constants.KAFKA_TOPICS)

    // Create context with 2 second batch interval
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("RealtimeRecommender")
    val ssc = new StreamingContext(sparkConf, Seconds(2))

    // Create direct kafka stream with brokers and topics
    val topicsSet = topics.split(",").toSet
    val kafkaParams = Map[String, String](
      "metadata.broker.list" -> brokers,
      "auto.offset.reset" -> "smallest")
    val messages = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topicsSet)

    messages.map(_._2).map{ event =>
      JSON.parseObject(event, classOf[NewClickEvent])
    }.mapPartitions { iter =>
      val jedis = RedisUtil.getJedis
      iter.map { event =>
        println("NewClickEvent" + event)
        val userId = event.asInstanceOf[NewClickEvent].getUserId
        val itemId = event.asInstanceOf[NewClickEvent].getItemId
        val key = "II:" + itemId
        val value = jedis.get(key)
        jedis.set("RUI:" + userId, value)
        println("Recommend to user:" + userId + ", items:" + value)
      }
    }.print()
    // Start the computation
    ssc.start()
    ssc.awaitTermination()
  }
}
