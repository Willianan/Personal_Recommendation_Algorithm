package com.dylan.BookCrossing;

import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class BXDataModelBuilder implements DataModelBuilder{
    @Override
    public DataModel buildDataModel(FastByIDMap<PreferenceArray> fastByIDMap) {
        return new GenericBooleanPrefDataModel(GenericBooleanPrefDataModel.toDataMap(fastByIDMap));
    }
}
