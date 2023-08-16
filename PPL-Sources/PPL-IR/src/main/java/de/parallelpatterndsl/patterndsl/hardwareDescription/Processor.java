package de.parallelpatterndsl.patterndsl.hardwareDescription;

import java.util.ArrayList;

public interface Processor {

    /**
     * Returns the unique (within a device) identifier for this team.
     * @return
     */
    public String getIdentifier();

    /**
     * Returns the number of available cores of the processor.
     * @return
     */
    public int getCores();

    /**
     * Returns the frequency of each core.
     * @return
     */
    public int getFrequency();

    /**
     * Returns the number of arithmetic units of each core.
     * @return
     */
    public int getArithmeticUnits();

    /**
     * Returns the vectorization supported by the device, if applicable.
     * @return
     */
    public String getVectorization();

    /**
     * Returns a list of the sizes of each cache, starting with the L1-cache.
     * @return
     */
    public ArrayList<Double> getCacheMemorySize();

    /**
     * Returns a list of the bandwidth of each cache, starting with the L1-cache.
     * @return
     */
    public ArrayList<Double> getCacheBandwidth();

    /**
     * Returns a list of the latency of each cache, starting with the L1-cache.
     * @return
     */
    public ArrayList<Double> getCacheLatency();

    /**
     * Returns the warp size of the device, if applicable.
     * @return
     */
    public int getWarpSize();

    /**
     * Returns the number of possible hyper-threads per core, if applicable.
     * @return
     */
    public int getHyperThreads();

    /**
     * Returns the device containing this processor.
     * @return
     */
    public Device getParent();

    /**
     * Returns the rank of the processor.
     * @return
     */
    public int getRank();

    /**
     * Tests if this Processor is the first in the parent device
     * @return
     */
    public boolean isFirstCG();
}
