package com.zkpk.reco.offline;

import org.apache.commons.io.Charsets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;

import java.io.*;
import java.util.regex.Pattern;

/**
 * Created by Dylan
 */
public class HDFSDataModel extends FileDataModel {
  private static final String COLON_DELIMTER = "::";
  private static final Pattern COLON_DELIMITER_PATTERN = Pattern.compile(COLON_DELIMTER);

  public HDFSDataModel(Configuration conf, String pathStr) throws IOException {
    this(conf, new Path(pathStr));
  }

  public HDFSDataModel(Configuration conf, Path path) throws IOException {
    super(storeHdfsFileToLocal(conf, path, COLON_DELIMTER));
  }

  private static File storeHdfsFileToLocal(Configuration conf, Path path, String delimiter) {
    // Now translate the file; remove commas, then convert "::" delimiter to comma
    File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "ratings.txt");
    if (resultFile.exists()) {
      resultFile.delete();
    }
    try{
      Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8);

      FileSystem fs = path.getFileSystem(conf);
      BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
      String line = br.readLine();
      while (line != null) {
        int lastDelimiterStart = line.lastIndexOf(COLON_DELIMTER);
        if (lastDelimiterStart < 0) {
          throw new IOException("Unexpected input format on line: " + line);
        }
        String subLine = line.substring(0, lastDelimiterStart);
        String convertedLine = COLON_DELIMITER_PATTERN.matcher(subLine).replaceAll(",");
        writer.write(convertedLine);
        writer.write('\n');
        line = br.readLine();
      }
    } catch(IOException e){
      e.printStackTrace();
    }
    return resultFile;
  }
}
