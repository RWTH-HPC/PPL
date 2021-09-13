package de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterParameter;
import de.se_rwth.commons.logging.Log;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.ArrayList;

/**
 * Implementation of the network interface based on a JSON input.
 */
public class Cluster implements Network {

    private JSONObject network;

    private ArrayList<Node> nodes;

    private ArrayList<ArrayList<Double>> connectivityBandwidthMap;

    private ArrayList<ArrayList<Double>> connectivityLatencyMap;

    private String topology;

    public Cluster(JSONObject network, String path) {
        this.network = network;
        nodes = new ArrayList<>();

        if (!network.containsKey(ClusterParameter.CONNECTIVITY_BANDWIDTH)) {
            Log.error("No connectivity matrix for bandwidth defined in network!");
        }
        connectivityBandwidthMap = getConnectivityBandwidthMap((JSONArray) network.get(ClusterParameter.CONNECTIVITY_BANDWIDTH));

        if (!network.containsKey(ClusterParameter.CONNECTIVITY_LATENCY)) {
            Log.error("No connectivity matrix for latency defined in network!");
        }
        connectivityLatencyMap = getConnectivityLatencyMap((JSONArray) network.get(ClusterParameter.CONNECTIVITY_LATENCY));

        if (!network.containsKey(ClusterParameter.TOPOLOGY)) {
            Log.error("No topology in network defined!");
        }
        this.topology = network.get(ClusterParameter.TOPOLOGY).toString();

        JSONArray jsonNodes = getJSONNodes(network);

        for (int i = 0; i < jsonNodes.size(); i++) {
            nodes.add(new Machine((JSONObject) jsonNodes.get(i),i,path));
        }

        if (nodes.size() != connectivityBandwidthMap.size()) {
            Log.error("Connectivity bandwidth is not of same size as nodes!");
        }

        if (nodes.size() != connectivityLatencyMap.size()) {
            Log.error("Connectivity latency is not of same size as nodes!");
        }
    }

    @Override
    public String getTopology() {
        return topology;
    }

    @Override
    public ArrayList<Node> getNodes() {
        return nodes;
    }

    @Override
    public double getConnectivityBandwidth(String node1, String node2) {
        if (node1.equals(node2)) {
            return 0;
        }

        int id1 = -1;
        int id2 = -1;

        for (int i = 0; i < nodes.size(); i++) {
            String name = nodes.get(i).getIdentifier();
            if (name.equals(node1)) {
                id1 = i;
            } else if(name.equals(node2)) {
                id2 = i;
            }
        }

        if (id1 == -1) {
            Log.error("Node-identifier: " + node1 + " does not exist!");
        }
        if (id2 == -1) {
            Log.error("Node-identifier: " + node2 + " does not exist!");
        }

        return connectivityBandwidthMap.get(id1).get(id2);
    }

    @Override
    public double getConnectivityLatency(String node1, String node2) {
        if (node1.equals(node2)) {
            return 0;
        }

        int id1 = -1;
        int id2 = -1;

        for (int i = 0; i < nodes.size(); i++) {
            String name = nodes.get(i).getIdentifier();
            if (name.equals(node1)) {
                id1 = i;
            } else if(name.equals(node2)) {
                id2 = i;
            }
        }

        if (id1 == -1) {
            Log.error("Node-identifier: " + node1 + " does not exist!");
        }
        if (id2 == -1) {
            Log.error("Node-identifier: " + node2 + " does not exist!");
        }

        return connectivityLatencyMap.get(id1).get(id2);
    }

    /**
     * Returns a JSONArray containing the node definitions.
     * @param network
     * @return
     */
    private JSONArray getJSONNodes(JSONObject network) {
        if (!network.containsKey(ClusterParameter.NODES)) {
            Log.error("No nodes in network defined!");
        }
        return (JSONArray) network.get(ClusterParameter.NODES);
    }

    /**
     * Returns the connectivity map for bandwidths, based on the JSONArray defining it.
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
