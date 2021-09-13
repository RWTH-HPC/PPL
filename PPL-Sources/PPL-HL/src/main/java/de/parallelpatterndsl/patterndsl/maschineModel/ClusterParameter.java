package de.parallelpatterndsl.patterndsl.maschineModel;

/**
 * Definitions of the necessary parameters for the JSON data.
 */
public class ClusterParameter {


    public static final String IDENTIFIER = "identifier";

    /*******************************************
     *
     *
     * Section: Network
     * Top-level object
     *
     ********************************************/

    public static final String TOPOLOGY = "topology";


    public static final String NODES = "nodes";

    /*******************************************
     *
     *
     * Section: Node
     *
     *
     ********************************************/

    public static final String TYPE = "type";

    public static final String DEVICES = "devices";

    public static final String ADDRESS = "address";

    public static final String TEMPLATE = "template";


    /*******************************************
     *
     *
     * Section: Device
     *
     *
     ********************************************/

    public static final String CACHE_GROUP = "cache-group";

    /*******************************************
     *
     *
     * Section: Processor
     *
     *
     ********************************************/
    public static final String CORES = "cores";

    public static final String FREQUENCY = "frequency";

    public static final String ARITHMETIC_UNITS = "arithmetic-units";

    public static final String WARP_SIZE = "warp-size";

    public static final String HYPER_THREADS = "hyper-threads";

    public static final String VECTORIZATION = "vectorization";

    public static final String CACHES = "caches";


    /*******************************************
     *
     *
     * Section: Metrics
     *
     *
     ********************************************/

    public static final String SIZE = "size";

    public static final String BANDWIDTH = "bandwidth";

    public static final String MAX_BANDWIDTH = "max-bandwidth";

    public static final String LATENCY = "latency";

    public static final String SHARING = "sharing";

    public static final String CONNECTIVITY_BANDWIDTH = "connectivity-bandwidth";

    public static final String CONNECTIVITY_LATENCY = "connectivity-latency";

}
