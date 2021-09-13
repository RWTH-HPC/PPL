package de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterParameter;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

public class CacheGroup implements Processor {

    private JSONObject cacheGroup;

    private Device parentDevice;

    private String identifier;

    private ArrayList<Double> cacheMemorySize;

    private ArrayList<Double> cacheBandwidth;

    private ArrayList<Double> cacheLatency;

    private int hyperThreads;

    private int cores;

    private int arithmeticUnits;

    private String vectorization;

    private int frequency;

    private int warpSize;

    private int rank;

    public CacheGroup(JSONObject processorJSON, Device parent, int rank, String path) {
        this.cacheGroup = processorJSON;
        this.rank = rank;

        cacheMemorySize = new ArrayList<>();
        cacheBandwidth = new ArrayList<>();
        cacheLatency = new ArrayList<>();

        this.parentDevice = parent;

        if (!cacheGroup.containsKey(ClusterParameter.IDENTIFIER)) {
            Log.error("No identifier in device defined!");
        }
        this.identifier = cacheGroup.get(ClusterParameter.IDENTIFIER).toString();

        // Handle a potential template argument.
        if (cacheGroup.containsKey(ClusterParameter.TEMPLATE)) {
            JSONParser parser = new JSONParser();


            try (Reader reader = new FileReader(path + cacheGroup.get(ClusterParameter.TEMPLATE).toString())) {

                cacheGroup = (JSONObject) parser.parse(reader);


            } catch (IOException e) {
                Log.error("Parsing failure! Not readable!");
                e.printStackTrace();
            } catch (ParseException e) {
                Log.error("Parsing failure! Not JSON format!");
                e.printStackTrace();
            }
        }

        // handle caches
        JSONArray caches = getCaches(cacheGroup);

        for (int i = 0; i < caches.size(); i++) {
            if (!((JSONObject) caches.get(i)).containsKey(ClusterParameter.SIZE)) {
                Log.error("No size in caches defined!");
            }
            cacheMemorySize.add(Double.parseDouble(((JSONObject) caches.get(i)).get(ClusterParameter.SIZE).toString()));
            if (!((JSONObject) caches.get(i)).containsKey(ClusterParameter.BANDWIDTH)) {
                Log.error("No bandwidth in caches defined!");
            }
            cacheBandwidth.add(Double.parseDouble(((JSONObject) caches.get(i)).get(ClusterParameter.BANDWIDTH).toString()));
            if (!((JSONObject) caches.get(i)).containsKey(ClusterParameter.LATENCY)) {
                Log.error("No latency in caches defined!");
            }
            cacheLatency.add(Double.parseDouble(((JSONObject) caches.get(i)).get(ClusterParameter.LATENCY).toString()));
        }


        if (!cacheGroup.containsKey(ClusterParameter.FREQUENCY)) {
            Log.error("No frequency in cache group defined!");
        }
        this.frequency = Integer.parseInt(cacheGroup.get(ClusterParameter.FREQUENCY).toString());

        if (cacheGroup.containsKey(ClusterParameter.ARITHMETIC_UNITS)) {
            this.arithmeticUnits = Integer.parseInt(cacheGroup.get(ClusterParameter.ARITHMETIC_UNITS).toString());
        }
        this.arithmeticUnits = 1;

        if (cacheGroup.containsKey(ClusterParameter.WARP_SIZE)) {
            this.warpSize = Integer.parseInt(cacheGroup.get(ClusterParameter.WARP_SIZE).toString());
        }
        this.warpSize = 1;


        if (cacheGroup.containsKey(ClusterParameter.HYPER_THREADS)) {
            this.hyperThreads = Integer.parseInt(cacheGroup.get(ClusterParameter.HYPER_THREADS).toString());
        }
        this.hyperThreads = 1;

        if (cacheGroup.containsKey(ClusterParameter.VECTORIZATION)) {
            this.vectorization = cacheGroup.get(ClusterParameter.VECTORIZATION).toString();
        }
        this.vectorization = "None";


        if (!cacheGroup.containsKey(ClusterParameter.CORES)) {
            Log.error("No cores in cache group defined!");
        }
        this.cores = Integer.parseInt(cacheGroup.get(ClusterParameter.CORES).toString());

    }

    @Override
    public String getIdentifier() {
        return identifier;
    }

    @Override
    public int getCores() {
        return cores;
    }

    @Override
    public int getFrequency() {
        return frequency;
    }

    @Override
    public int getArithmeticUnits() {
        return arithmeticUnits;
    }

    @Override
    public String getVectorization() {
        return vectorization;
    }

    @Override
    public ArrayList<Double> getCacheMemorySize() {
        return cacheMemorySize;
    }

    @Override
    public ArrayList<Double> getCacheBandwidth() {
        return cacheBandwidth;
    }

    @Override
    public ArrayList<Double> getCacheLatency() {
        return cacheLatency;
    }

    @Override
    public int getWarpSize() {
        return warpSize;
    }

    @Override
    public int getHyperThreads() {
        return hyperThreads;
    }

    @Override
    public Device getParent() {
        return parentDevice;
    }

    @Override
    public int getRank() {
        return rank;
    }


    private JSONArray getCaches(JSONObject cacheGroup) {
        if (!cacheGroup.containsKey(ClusterParameter.CACHES)) {
            Log.error("No caches in cache group defined!");
        }
        return (JSONArray) cacheGroup.get(ClusterParameter.CACHES);
    }
}
