
package com.dylan.recom.offline;

import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.precompute.MultithreadedBatchItemSimilarities;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;

import java.io.File;

/**
 * generate item-item similarity table & user-item table, and insert into redis
 *
 */
public final class SimilarityTablesGenerator {

  private SimilarityTablesGenerator() {}

  public static void main(String[] args) throws Exception {
/*
    if (args.length != 1) {
      System.err.println("Need path to ratings.dat of the movielens1M dataset as argument!");
      System.exit(-1);
    }
*/
    DataModel dataModel = new GroupLensDataModel();
    UserItemSimilarityTableRedisWriter userItemSimilarityTableRedisWriter =
        new UserItemSimilarityTableRedisWriter(dataModel);
    userItemSimilarityTableRedisWriter.storeToRedis();

    ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel,
        new LogLikelihoodSimilarity(dataModel));
    BatchItemSimilarities batch = new MultithreadedBatchItemSimilarities(recommender, 5);

    int numSimilarities = batch.computeItemSimilarities(Runtime.getRuntime().availableProcessors(), 1,
        new ItemsSimilarityTableRedisWriter());

    System.out.println("Computed " + numSimilarities + " similarities for " + dataModel.getNumItems() + " items "
        + "and saved them to redis");

    userItemSimilarityTableRedisWriter.waitUtilDone();
  }

}
