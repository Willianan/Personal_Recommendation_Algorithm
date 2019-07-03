package com.dylan.recom.realtime;

/**
 * Created by dylan
 */

import com.alibaba.fastjson.JSON;
import com.dylan.recom.common.Constants;
import org.apache.log4j.Logger;

import java.util.Properties;

import kafka.javaapi.producer.Producer;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;

public class KafkaProducer implements Runnable {
  private static final Logger LOGGER = Logger.getLogger(KafkaProducer.class);
  private final String topic;

  public KafkaProducer(String topic) {
    this.topic = topic;
  }

  static NewClickEvent[] newClickEvents = new NewClickEvent[]{
      new NewClickEvent(1000000L, 123L),
      new NewClickEvent(1000001L, 111L),
      new NewClickEvent(1000002L, 500L),
      new NewClickEvent(1000003L, 278L),
      new NewClickEvent(1000004L, 681L),
  };

  public void run() {
    Properties props = new Properties();
    props.put("metadata.broker.list", Constants.KAFKA_ADDR);
    props.put("serializer.class", "kafka.serializer.StringEncoder");
    props.put("producer.type", "async");
    ProducerConfig conf = new ProducerConfig(props);
    Producer<Integer, String> producer = null;
    try {
      System.out.println("Producing messages");
      producer = new Producer<>(conf);
      for (NewClickEvent event : newClickEvents) {
        String eventAsStr = JSON.toJSONString(event);
        producer.send(new KeyedMessage<Integer, String>(
            this.topic, eventAsStr));
        System.out.println("Sending messages:" + eventAsStr);

      }
      System.out.println("Done sending messages");
    } catch (Exception ex) {
      LOGGER.fatal("Error while producing messages", ex);
      LOGGER.trace(null, ex);
      System.err.println("Error while producing messages：" + ex);
    } finally {
      if (producer != null) producer.close();
    }
  }

  public static void main(String[] args) throws Exception {
    new Thread(new KafkaProducer(Constants.KAFKA_TOPICS)).start();
  }
}
