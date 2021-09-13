package de.parallelpatterndsl.patterndsl.hardwareDescription;

import java.util.ArrayList;

/**
 * Interface definition of a device within a node.
 */
public interface Device {

    /**
     * Returns the type of device CPU, GPU etc...
     * @return
     */
    public String getType();

    /**
     * Returns an identifier of the device.
     * @return
     */
    public String getIdentifier();

    /**
     * Returns the processors of the device.
     * @return
     */
    public ArrayList<Processor> getProcessor();


    /**
     * Returns the size of the main memory connected to this device.
     * @return
     */
    public double getMainMemorySize();

    /**
     * Return the bandwidth to the main memory connected to this device.
     * @return
     */
    public double getMainBandwidth();

    /**
     * Return the latency to the main memory connected to this device.
     * @return
     */
    public double getMainLatency();

    /**
     * Return the maximum bandwidth to the main memory connected to this device can achieve.
     * @return
     */
    public double getMaxMainBandwidth();

    /**
     * Returns the node holding this device.
     * @return
     */
    public Node getParent();

    /**
     * Returns the number of the corresponding GPU in Cuda.
     * @return
     */
    public int getGPUrank();

}
