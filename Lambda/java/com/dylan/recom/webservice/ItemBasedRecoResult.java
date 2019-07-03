package com.dylan.recom.webservice;

/**
 * Created by dylan
 */
import com.alibaba.fastjson.JSON;
import com.dylan.recom.common.ItemSimilarity;
import com.dylan.recom.common.RedisUtil;
import redis.clients.jedis.Jedis;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import java.util.*;

@Path("/ws/v1/recom")
public class ItemBasedRecoResult {
  Jedis jedis = null;

  public ItemBasedRecoResult() {
    jedis = RedisUtil.getJedis();
  }

  @GET
  @Path("/{userid}")
  @Produces(MediaType.APPLICATION_JSON)
  public RecommendedItems getRecoItems(@PathParam("userid") String userid) {
    RecommendedItems recommendedItems = new RecommendedItems();

    // Stage 1: get user's items
    String key = String.format("UI:%s", userid);
    String value = jedis.get(key);
    if(value == null || value.length() <= 0) {
      return recommendedItems;
    }

    List<Long> userItems = JSON.parseArray(value, Long.class);
    Set<Long> userItemsSet = new TreeSet<Long>(userItems);

    // Stage 2: get similar items to the user's items
    List<String> userItemStrs = new ArrayList<>();
    for(Long item: userItems) {
      userItemStrs.add("II:" + item);
    }

    List<String> similarItems = jedis.mget(userItemStrs.toArray(new String[userItemStrs.size()]));
    Set<ItemSimilarity> similarItemsSet = new TreeSet<>();
    for(String item: similarItems) {
      List<ItemSimilarity> result = JSON.parseArray(item, ItemSimilarity.class);
      similarItemsSet.addAll(result);
    }

    List<Long> recommendedItemIDs = new ArrayList<>();
    for(ItemSimilarity item: similarItemsSet) {
      if(!userItemsSet.contains(item.getId())) {
        recommendedItemIDs.add((item.getId()));
      }
      if(recommendedItemIDs.size() >= 10)
        break;
    }
    recommendedItems.setItems(recommendedItemIDs.toArray(new Long[0]));
    return recommendedItems;
  }
}
