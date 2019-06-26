package com.dylan.MovieLens;

import org.apache.commons.io.Charsets;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

import java.io.*;
import java.util.regex.Pattern;

public class MovieLensDataModel extends FileDataModel {

    private static String COLON_DELIMITER="::";
    private static Pattern COLON_DELIMITER_PATTERN=Pattern.compile(COLON_DELIMITER);

    public MovieLensDataModel(File ratingsFile) throws IOException{
        super(convertFile(ratingsFile));
    }

    private static File convertFile(File orginalFile) throws IOException{
        File resultFile = new File(System.getProperty("java.io.tmpdir"), "ratings.csv");
        if (resultFile.exists()){
            resultFile.delete();
        }
        try(Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8)) {

            for (String line: new FileLineIterable(orginalFile, false)){
                int lastIndex = line.lastIndexOf(COLON_DELIMITER);

                if (lastIndex < 0 ){
                    throw new IOException("Invalid data!");
                }
                String subLine = line.substring(0, lastIndex);

                String convertedSubLine = COLON_DELIMITER_PATTERN.matcher(subLine).replaceAll(",");
                writer.write(convertedSubLine);
                writer.write('\n');
            }
        } catch (IOException ioe){
            resultFile.delete();
            throw ioe;
        }
        return resultFile;
    }
}
