package com.dylan.recom.offline;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.alibaba.fastjson.JSON;
import com.dylan.recom.common.ItemSimilarity;
import com.dylan.recom.common.RedisUtil;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItem;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItems;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;
import redis.clients.jedis.Jedis;

// generate item-item similarity table
public class ItemsSimilarityTableRedisWriter implements SimilarItemsWriter {
  private long itemCounter = 0;
  private Jedis jedis = null;
  @Override
  public void open() throws IOException {
    jedis = RedisUtil.getJedis();
  }

  @Override
  public void add(SimilarItems similarItems) throws IOException {
    ItemSimilarity[] values = new ItemSimilarity[similarItems.numSimilarItems()];
    int counter = 0;
    for (SimilarItem item: similarItems.getSimilarItems()) {
      values[counter] = new ItemSimilarity(item.getItemID(), item.getSimilarity());
      counter++;
    }
    String key = "II:" + similarItems.getItemID();
    String items = JSON.toJSONString(values);
    jedis.set(key, items);
    itemCounter++;
    if(itemCounter % 100 == 0) {
      System.out.println("Store " + key + " to redis, total:" + itemCounter);
    }
  }

  @Override
  public void close() throws IOException {
    jedis.close();
  }
}

