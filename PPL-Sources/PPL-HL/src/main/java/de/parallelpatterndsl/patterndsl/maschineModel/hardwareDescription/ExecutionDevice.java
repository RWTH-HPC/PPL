package de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterParameter;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

/**
 * Implementation of the device interface based on a JSON input.
 */
public class ExecutionDevice implements Device {

    private JSONObject device;

    private Machine parentNode;

    private String type;

    private String identifier;

    private double mainMemorySize;

    private double mainMemoryBandwidth;

    private double mainMemoryLatency;

    private double maxMainMemoryBandwidth;

    private ArrayList<Processor> cacheGroups;

    private int GPUrank;

    public ExecutionDevice(JSONObject deviceJSON, Machine parent, String path) {
        this.device = deviceJSON;

        this.parentNode = parent;

        cacheGroups = new ArrayList<>();

        if (!device.containsKey(ClusterParameter.IDENTIFIER)) {
            Log.error("No identifier in device defined!");
        }
        this.identifier = device.get(ClusterParameter.IDENTIFIER).toString();

        // Handle a potential template argument.
        if (device.containsKey(ClusterParameter.TEMPLATE)) {
            JSONParser parser = new JSONParser();


            try (Reader reader = new FileReader(path + device.get(ClusterParameter.TEMPLATE).toString())) {

                path = path + device.get(ClusterParameter.TEMPLATE).toString();
                File file = new File(path);
                path = file.getAbsolutePath().substring(0,file.getAbsolutePath().length() - file.getName().length());

                device = (JSONObject) parser.parse(reader);


            } catch (IOException e) {
                Log.error("Parsing failure! Not readable! " + path + device.get(ClusterParameter.TEMPLATE).toString());
                e.printStackTrace();
            } catch (ParseException e) {
                Log.error("Parsing failure! No JSON File in:" + path + device.get(ClusterParameter.TEMPLATE).toString());
                e.printStackTrace();
            }
        }

        JSONArray jsonCacheGroups = getCacheGroups(device);
        for (int i = 0; i < jsonCacheGroups.size(); i++) {
            cacheGroups.add(new CacheGroup((JSONObject) jsonCacheGroups.get(i), this, i, path));
        }

        if (!device.containsKey(ClusterParameter.TYPE)) {
            Log.error("No type in device defined!");
        }
        this.type = device.get(ClusterParameter.TYPE).toString();


        if (!device.containsKey(ClusterParameter.SIZE)) {
            Log.error("No main memory size in device defined!");
        }
        this.mainMemorySize = Double.parseDouble(device.get(ClusterParameter.SIZE).toString());

        if (!device.containsKey(ClusterParameter.BANDWIDTH)) {
            Log.error("No main memory bandwidth in device defined!");
        }
        this.mainMemoryBandwidth = Double.parseDouble(device.get(ClusterParameter.BANDWIDTH).toString());

        if (!device.containsKey(ClusterParameter.LATENCY)) {
            Log.error("No main memory latency in device defined!");
        }
        this.mainMemoryLatency = Double.parseDouble(device.get(ClusterParameter.LATENCY).toString());

        if (!device.containsKey(ClusterParameter.MAX_BANDWIDTH)) {
            Log.error("No main memory bandwidth in device defined!");
        }
        this.maxMainMemoryBandwidth = Double.parseDouble(device.get(ClusterParameter.MAX_BANDWIDTH).toString());

    }

    @Override
    public String getType() {
        return type;
    }

    @Override
    public String getIdentifier() {
        return identifier;
    }

    @Override
    public ArrayList<Processor> getProcessor() {
        return cacheGroups;
    }

    @Override
    public double getMainMemorySize() {
        return mainMemorySize;
    }

    @Override
    public double getMainBandwidth() {
        return mainMemoryBandwidth;
    }

    @Override
    public double getMainLatency() {
        return mainMemoryLatency;
    }

    @Override
    public double getMaxMainBandwidth() {
        return maxMainMemoryBandwidth;
    }

    @Override
    public Node getParent() {
        return parentNode;
    }

    @Override
    public int getGPUrank() {
        return GPUrank;
    }

    public void setGPUrank(int GPUrank) {
        this.GPUrank = GPUrank;
    }

    private JSONArray getCacheGroups(JSONObject cacheGroup) {
        if (!cacheGroup.containsKey(ClusterParameter.CACHE_GROUP)) {
            Log.error("No caches in cache group defined!");
        }
        return (JSONArray) cacheGroup.get(ClusterParameter.CACHE_GROUP);
    }


}
