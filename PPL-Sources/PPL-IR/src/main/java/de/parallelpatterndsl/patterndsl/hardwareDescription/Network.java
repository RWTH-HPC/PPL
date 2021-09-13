package de.parallelpatterndsl.patterndsl.hardwareDescription;

import java.util.ArrayList;

/**
 * Interface definition of the network.
 */
public interface Network {

    /**
     * Returns an identifier to describe the network topology of the network running the code.
     * @return
     */
    public String getTopology();

    /**
     * Returns a list of nodes available within the network.
     * @return
     */
    public ArrayList<Node> getNodes();

    /**
     * Returns the bandwidth between node1 and node2.
     * @param node1 identifier of the first node
     * @param node2 identifier of the second node
     * @return
     */
    public double getConnectivityBandwidth(String node1, String node2);

    /**
     * Returns the latency between node1 and node2.
     * @param node1 identifier of the first node
     * @param node2 identifier of the second node
     * @return
     */
    public double getConnectivityLatency(String node1, String node2);
}
