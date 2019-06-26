package com.dylan.MovieLens;


import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.precompute.FileSimilarItemsWriter;
import org.apache.mahout.cf.taste.impl.similarity.precompute.MultithreadedBatchItemSimilarities;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;

import java.io.File;

/**
 * Example that precomputes all item similarities of the Movielens1M dataset
 *
 * Usage: download movielens1M from http://www.grouplens.org/node/73 , unzip it and invoke this code with the path
 * to the ratings.dat file as argument
 *
 */
public final class BatchItemSimilaritiesGroupLens {

    private BatchItemSimilaritiesGroupLens() {}

    public static void main(String[] args) throws Exception {

        if (args.length != 1) {
            System.err.println("Need path to ratings.dat of the movielens1M dataset as argument!");
            System.exit(-1);
        }

        File resultFile = new File(System.getProperty("java.io.tmpdir"), "similarities.csv");
        if (resultFile.exists()) {
            resultFile.delete();
        }

        DataModel dataModel = new GroupLensDataModel(new File(args[0]));
        ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel,
                new LogLikelihoodSimilarity(dataModel));
        BatchItemSimilarities batch = new MultithreadedBatchItemSimilarities(recommender, 5);

        int numSimilarities = batch.computeItemSimilarities(Runtime.getRuntime().availableProcessors(), 1,
                new FileSimilarItemsWriter(resultFile));

        System.out.println("Computed " + numSimilarities + " similarities for " + dataModel.getNumItems() + " items "
                + "and saved them to " + resultFile.getAbsolutePath());
    }

}