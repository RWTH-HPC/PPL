package de.parallelpatterndsl.patterndsl.MappingTree.JSONReader;

import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class JSONAMTParser {
    /**
     * Reads the AMT from a JSON file
     * @param path
     * @return
     */
    public static JSONObject readAMTJSON(String path) {
        JSONParser parser = new JSONParser();

        try (Reader reader = new FileReader(path)) {

            return (JSONObject) parser.parse(reader);

        } catch (IOException e) {
            Log.error("Parsing failure! No JSON File in:" + path);
            e.printStackTrace();
        } catch (ParseException e) {
            Log.error("Parsing failure! Not JSON format!" + path);
            e.printStackTrace();
        }

        return null;
    }
}
