package com.dylan.example;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.*;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.*;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

import java.io.File;

public class IREvaluatorIntro {
    private IREvaluatorIntro() {
    }

    public static void main(String[] args) throws Exception {

        RandomUtils.useTestSeed();

        final DataModel model = new FileDataModel(new File("/root/data/ua.base"));
        RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();

        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model) throws TasteException {
                UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };

        IRStatistics stats = evaluator.evaluate(recommenderBuilder, null, model, null, 5, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);

        System.out.println(stats.getPrecision());
        System.out.println(stats.getRecall());
        System.out.println(stats.getF1Measure());
    }
}
