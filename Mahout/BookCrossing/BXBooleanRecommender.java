package com.dylan.BookCrossing;

import com.sun.tools.internal.xjc.reader.xmlschema.bindinfo.BIConversion;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.CachingUserSimilarity;

import java.util.Collection;
import java.util.List;

public class BXBooleanRecommender implements Recommender{
    private Recommender recommender;
    public BXBooleanRecommender(DataModel dataModel) throws TasteException{
        UserSimilarity similarity = new CachingUserSimilarity(new LogLikelihoodSimilarity(dataModel), dataModel);
        //UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, Double.NEGATIVE_INFINITY, similarity,dataModel, 1.0);
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.5, similarity, dataModel, 1.0);
        recommender = new GenericBooleanPrefUserBasedRecommender(dataModel, neighborhood, similarity);
    }

    public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
        return recommender.recommend(userID, howMany, (IDRescorer) null, false);
    }

    public List<RecommendedItem> recommend(long userID, int howMany, boolean includeKnownItems) throws TasteException {
        return recommender.recommend(userID, howMany, (IDRescorer) null, includeKnownItems);
    }

    public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
        return recommender.recommend(userID, howMany, rescorer, false);
    }

    @Override
    public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer idRescorer, boolean includeKnownItems) throws TasteException {
        return recommender.recommend(userID, howMany, (IDRescorer) null, includeKnownItems);
    }

    @Override
    public float estimatePreference(long userID, long itemID) throws TasteException {
        return recommender.estimatePreference(userID, itemID);
    }

    public void setPreference(long userID, long itemID, float value) throws TasteException {
        recommender.setPreference(userID, itemID, value);
    }

    public void removePreference(long userID, long itemID) throws TasteException {
        recommender.removePreference(userID, itemID);
    }

    public DataModel getDataModel() {
        return recommender.getDataModel();
    }

    @Override
    public void refresh(Collection<Refreshable> collection) {
        recommender.refresh(collection);
    }
}
