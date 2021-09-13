package de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterParameter;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

/**
 * Implementation of the node interface based on a JSON input.
 */
public class Machine implements Node {

    private JSONObject node;

    private ArrayList<Device> devices;

    private String type;

    private String identifier;

    private String address;

    private ArrayList<ArrayList<Double>> connectivityBandwidthMap;

    private ArrayList<ArrayList<Double>> connectivityLatencyMap;

    private int rank;

    public Machine(JSONObject nodeObject, int rank, String path) {
        this.node = nodeObject;
        this.rank = rank;
        devices = new ArrayList<>();

        if (!node.containsKey(ClusterParameter.IDENTIFIER)) {
            Log.error("No type in node defined!");
        }
        this.identifier = node.get(ClusterParameter.IDENTIFIER).toString();

        if (!node.containsKey(ClusterParameter.ADDRESS)) {
            Log.error("No type in node defined!");
        }
        this.address = node.get(ClusterParameter.ADDRESS).toString();


        // Handle templates for nodes.
        if (node.containsKey(ClusterParameter.TEMPLATE)) {
            JSONParser parser = new JSONParser();


            try (Reader reader = new FileReader(path + node.get(ClusterParameter.TEMPLATE).toString())) {

                path = path + node.get(ClusterParameter.TEMPLATE).toString();
                File file = new File(path);
                path = file.getAbsolutePath().substring(0,file.getAbsolutePath().length() - file.getName().length());

                node = (JSONObject) parser.parse(reader);



            } catch (IOException e) {
                Log.error("Parsing failure! Not readable!");
                e.printStackTrace();
            } catch (ParseException e) {
                Log.error("Parsing failure! Not JSON format!");
                e.printStackTrace();
            }
        }

        if (!node.containsKey(ClusterParameter.CONNECTIVITY_BANDWIDTH)) {
            Log.error("No connectivity matrix for bandwidth defined in network!");
        }
        connectivityBandwidthMap = getConnectivityBandwidthMap((JSONArray) node.get(ClusterParameter.CONNECTIVITY_BANDWIDTH));

        if (!node.containsKey(ClusterParameter.CONNECTIVITY_LATENCY)) {
            Log.error("No connectivity matrix for latency defined in network!");
        }
        connectivityLatencyMap = getConnectivityLatencyMap((JSONArray) node.get(ClusterParameter.CONNECTIVITY_LATENCY));

        JSONArray jsonDevices = getJSONDevices(node);
        for (int i = 0; i < jsonDevices.size(); i++) {
            devices.add(new ExecutionDevice((JSONObject) jsonDevices.get(i), this, path));
        }

        if (!node.containsKey(ClusterParameter.TYPE)) {
            Log.error("No type in node defined!");
        }
        this.type = node.get(ClusterParameter.TYPE).toString();


        if (devices.size() != connectivityBandwidthMap.size()) {
            Log.error("Connectivity bandwidth is not of same size as devices!");
        }

        if (devices.size() != connectivityLatencyMap.size()) {
            Log.error("Connectivity latency is not of same size as devices!");
        }

        int currentGPURank = -1;
        for (Device device: devices) {
            if (device.getType().equalsIgnoreCase("gpu") && device instanceof ExecutionDevice); {
                ((ExecutionDevice) device).setGPUrank(currentGPURank);
                currentGPURank++;
            }
        }
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
    public String getAddress() {
        return address;
    }

    @Override
    public double getConnectivityBandwidth(String device1, String device2) {
        if (device1.equals(device2)) {
            return 0;
        }

        int id1 = -1;
        int id2 = -1;

        for (int i = 0; i < devices.size(); i++) {
            String name = devices.get(i).getIdentifier();
            if (name.equals(device1)) {
                id1 = i;
            } else if (name.equals(device2)) {
                id2 = i;
            }
        }

        if (id1 == -1) {
            Log.error("Node-identifier: " + device1 + " does not exist!");
        }
        if (id2 == -1) {
            Log.error("Node-identifier: " + device2 + " does not exist!");
        }

        return connectivityBandwidthMap.get(id1).get(id2);
    }

    @Override
    public double getConnectivityLatency(String device1, String device2) {
        if (device1.equals(device2)) {
            return 0;
        }

        int id1 = -1;
        int id2 = -1;

        for (int i = 0; i < devices.size(); i++) {
            String name = devices.get(i).getIdentifier();
            if (name.equals(device1)) {
                id1 = i;
            } else if (name.equals(device2)) {
                id2 = i;
            }
        }

        if (id1 == -1) {
            Log.error("Node-identifier: " + device1 + " does not exist!");
        }
        if (id2 == -1) {
            Log.error("Node-identifier: " + device2 + " does not exist!");
        }

        return connectivityLatencyMap.get(id1).get(id2);
    }

    @Override
    public ArrayList<Device> getDevices() {
        return devices;
    }

    @Override
    public int getRank() {
        return rank;
    }


    private JSONArray getJSONDevices(JSONObject machine) {
        if (!machine.containsKey(ClusterParameter.DEVICES)) {
            Log.error("No devices in node defined!");
        }
        return (JSONArray) machine.get(ClusterParameter.DEVICES);
    }

    /**
     * Returns the connectivity map for bandwidths, based on the JSONArray defining it.
     *
     * @param JSONBandMap
     * @return
     */
    private ArrayList<ArrayList<Double>> getConnectivityBandwidthMap(JSONArray JSONBandMap) {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();

        for (int i = 0; i < JSONBandMap.size(); i++) {
            JSONArray intermediateJSON = (JSONArray) JSONBandMap.get(i);
            ArrayList<Double> intermediateArray = new ArrayList<>();

            for (int j = 0; j < intermediateJSON.size(); j++) {
                intermediateArray.add(Double.parseDouble(intermediateJSON.get(j).toString()));
            }

            result.add(intermediateArray);
        }

        return result;
    }

    /**
     * Returns the connectivity map for latencies, based on the JSONArray defining it.
     *
     * @param JSONLatMap
     * @return
     */
    private ArrayList<ArrayList<Double>> getConnectivityLatencyMap(JSONArray JSONLatMap) {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();

        for (int i = 0; i < JSONLatMap.size(); i++) {
            JSONArray intermediateJSON = (JSONArray) JSONLatMap.get(i);
            ArrayList<Double> intermediateArray = new ArrayList<>();

            for (int j = 0; j < intermediateJSON.size(); j++) {
                intermediateArray.add(Double.parseDouble(intermediateJSON.get(j).toString()));
            }

            result.add(intermediateArray);
        }

        return result;
    }

}
