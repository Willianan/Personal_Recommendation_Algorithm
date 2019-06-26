package com.dylan.example;

import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class CreatePreferenceArray {
    private CreatePreferenceArray() {
    }

    public static void main(String[] args) {
        PreferenceArray User1Pref = new GenericUserPreferenceArray(2);
        User1Pref.setUserID(0, 1L);
        User1Pref.setItemID(0, 101L);
        User1Pref.setValue(0, 3.0f);
        User1Pref.setItemID(1, 102L);
        User1Pref.setValue(1, 4.0f);
        Preference pref = User1Pref.get(1);
        System.out.println(User1Pref);
    }
}
