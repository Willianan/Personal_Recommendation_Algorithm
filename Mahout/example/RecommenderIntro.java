package com.dylan.example;

import org.apache.mahout.cf.taste.impl.model.file.*;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.impl.neighborhood.*;
import org.apache.mahout.cf.taste.impl.recommender.*;

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.*;
import org.apache.mahout.cf.taste.neighborhood.*;
import org.apache.mahout.cf.taste.recommender.*;

import java.io.File;
import java.util.List;

public class RecommenderIntro {
    private RecommenderIntro() {
    }

    public static void main(String[] args) throws Exception{
        DataModel model = new FileDataModel(new File("/root/data/ua.base"));
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, model);
        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        List<RecommendedItem> recommendedItems = recommender.recommend(1, 20);

        for (RecommendedItem recommendedItem: recommendedItems){
            System.out.println(recommendedItem);
        }
    }
}
