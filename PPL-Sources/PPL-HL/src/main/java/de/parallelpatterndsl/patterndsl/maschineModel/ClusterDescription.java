package de.parallelpatterndsl.patterndsl.maschineModel;

import de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription.Cluster;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import de.se_rwth.commons.logging.Log;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

/**
 * JSON parser for the cluster description.
 */
public class ClusterDescription {

    public static Cluster cluster;

    private ClusterDescription() {
    }

    public static Cluster parse(String path) {
        ClusterDescription description = new ClusterDescription();

        JSONParser parser = new JSONParser();

        try (Reader reader = new FileReader(path)) {

            JSONObject jsonObject = (JSONObject) parser.parse(reader);

            File file = new File(path);
            String mainPath = file.getAbsolutePath().substring(0,file.getAbsolutePath().length() - file.getName().length());
            cluster = new Cluster(jsonObject,mainPath);



        } catch (IOException e) {
            Log.error("Parsing failure! File not found!");
            e.printStackTrace();
        } catch (ParseException e) {
            Log.error("Parsing failure! Not JSON format!");
            e.printStackTrace();
        }

        return cluster;
    }


}
