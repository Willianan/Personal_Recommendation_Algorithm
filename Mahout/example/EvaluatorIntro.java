package com.dylan.example;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
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

public class EvaluatorIntro {
    private EvaluatorIntro() {
    }

    public static void main(String[] args) throws Exception {

        RandomUtils.useTestSeed();

        final DataModel model = new FileDataModel(new File("/root/data/ua.base"));
        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        RecommenderEvaluator recommenderEvaluator = new RMSRecommenderEvaluator();

        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model) throws TasteException {
                UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };

        double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);
        double rmse = recommenderEvaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);

        System.out.println(score);
        System.out.println(rmse);
    }
}
