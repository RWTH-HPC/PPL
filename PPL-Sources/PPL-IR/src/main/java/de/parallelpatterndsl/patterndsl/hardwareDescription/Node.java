package de.parallelpatterndsl.patterndsl.hardwareDescription;

import java.util.ArrayList;

/**
 * Interface for the hardware description of a set of identical nodes.
 */
public interface Node {

    /**
     * Returns an identifier for the type of a node e.g. NUMA etc..
     * @return
     */
    public String getType();

    /**
     * Returns an identifier for the node.
     * @return
     */
    public String getIdentifier();

    /**
     * Returns the address for the node.
     * @return
     */
    public String getAddress();


    /**
     * Returns the bandwidth between device1 and device2.
     * @param device1 identifier of the first device
     * @param device2 identifier of the second device
     * @return
     */
    public double getConnectivityBandwidth(String device1, String device2);

    /**
     * Returns the latency between device1 and device2.
     * @param device1 identifier of the first device
     * @param device2 identifier of the second device
     * @return
     */
    public double getConnectivityLatency(String device1, String device2);

    /**
     * Returns a list of all devices integrated into the node e.g. CPUs, GPUs etc...
     * @return
     */
    public ArrayList<Device> getDevices();

    /**
     * Returns the rank of the machine.
     * @return
     */
    public int getRank();

}
