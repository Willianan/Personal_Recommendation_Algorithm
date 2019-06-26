package com.dylan.practice;

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class MysqlDataMovieRecommend {
    private MysqlDataMovieRecommend() throws TasteException, IOException {
    }

    public static void main(String[] args) throws TasteException, IOException {
        File resultFile = new File("/tmp", "MysqlMovieRcomed.txt");
        //Mysql Connection
        MysqlDataSource mysqlDataSource = new MysqlDataSource();
        mysqlDataSource.setDatabaseName("mahout");
        mysqlDataSource.setServerName("127.0.0.1");
        mysqlDataSource.setUser("mahout");
        mysqlDataSource.setPassword("mahout");
        mysqlDataSource.setAutoReconnect(true);
        mysqlDataSource.setFailOverReadOnly(false);


        JDBCDataModel dataModel = new MySQLJDBCDataModel(mysqlDataSource, "taste_preferences2", "user_id", "item_id", "preference", null);
        DataModel model = dataModel;

        //Recommendations
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        //UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.5, similarity, model, 1.0);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        try (PrintWriter writer = new PrintWriter(resultFile)) {
            for (int userID = 1; userID <= model.getNumUsers(); userID++) {
                List<RecommendedItem> recommendedItems = recommender.recommend(userID, 3);
                String line = userID + " : ";
                for (RecommendedItem recommendedItem : recommendedItems) {
                    line += recommendedItem.getItemID() + ":" + recommendedItem.getValue() + ",";
                }
                if (line.endsWith(",")) {
                    line = line.substring(0, line.length() - 1);
                }
                writer.write(line);
                writer.write('\n');
            }
        } catch (IOException ioe) {
            resultFile.delete();
            throw ioe;
        }
        System.out.println("Recommended for " + model.getNumUsers() + " users and saved them to " + resultFile.getAbsolutePath());
    }
}

