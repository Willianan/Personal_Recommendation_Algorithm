package com.dylan.MovieLens;


import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class UserBasedRecommenderMovieLens {
    private UserBasedRecommenderMovieLens() {
    }

    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            System.err.println("Needs MovieLens 1M dataset as argument!");
            System.exit(-1);
        }

        File resultFile = new File(System.getProperty("java.io.tmpdir"), "userRecom.csv");
        if (resultFile.exists()) {
            resultFile.delete();
        }

        final DataModel dataModel = new GroupLensDataModel(new File(args[0]));
        UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);

        Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
        final Recommender cachingRecommender = new CachingRecommender(recommender);

        //Evaluate
        RMSRecommenderEvaluator rmsRecommenderEvaluator = new RMSRecommenderEvaluator();
        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
                return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
            }
        };
        double score = rmsRecommenderEvaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
        System.out.println("RMSE score is " +score);

        try(PrintWriter writer = new PrintWriter(resultFile, "UTF-8")) {
            for (int userID = 1; userID <= dataModel.getNumUsers(); userID++){
                List<RecommendedItem> recommendedItems = cachingRecommender.recommend(userID, 2);
                String line = userID +":";
                for (RecommendedItem recommendedItem : recommendedItems){
                    line += recommendedItem.getItemID()+"|"+recommendedItem.getValue()+",";
                }
                if (line.endsWith(",")){
                    line = line.substring(0, line.length()-1);
                }
                writer.write(line);
                writer.write('\n');
            }
            //writer.close();
        } catch (IOException ioe) {
            resultFile.delete();
            throw ioe;
        }
        System.out.println("Recommended for "+dataModel.getNumUsers() + " users "
                + "and saved them to " + resultFile.getAbsolutePath());
    }
}