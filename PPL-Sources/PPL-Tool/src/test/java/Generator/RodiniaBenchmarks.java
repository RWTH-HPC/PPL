package Generator;

import java.util.HashMap;

/**
 * Enum describing all test cases for a parameterized test.
 * The Cases are encoded in the following way:
 */
public enum RodiniaBenchmarks {


    Backpropagation,
    Heartwall,
    Stream,
    CFD,
    Lud,
    Breadth_First_Search,
    Hotspot,
    Hotspot3D,
    Kmeans,
    LavaMD,
    Leukocyte_pre,
    Myocyte,
    Neighbor_pre,
    Neighbor,
    Particle,
    Pathfinder,
    Srad,
    Needle,
    Leukocyte,
    //Parsing
    ;


    private static final String globalPath = "../../Samples/rodinia-ppl/";

    public static final HashMap<RodiniaBenchmarks, TestCase> paths;
    static {
        paths = new HashMap<>();
        paths.put(Backpropagation, new TestCase(globalPath + "Back-Propagation/", "backprop"));
        paths.put(Breadth_First_Search, new TestCase(globalPath + "Breadth-First-Search/", "bfs"));
        paths.put(CFD, new TestCase(globalPath + "CFD-Solver/", "cfd"));
        paths.put(Heartwall, new TestCase(globalPath + "Heartwall/", "heartwall"));
        paths.put(Hotspot, new TestCase(globalPath + "Hotspot/", "hotspot"));
        paths.put(Hotspot3D, new TestCase(globalPath + "Hotspot3D/", "hotspot3D"));
        paths.put(Kmeans, new TestCase(globalPath + "Kmeans/", "Kmeans"));
        paths.put(LavaMD, new TestCase(globalPath + "LavaMD/", "LavaMD"));
        paths.put(Lud, new TestCase(globalPath + "lud/", "lud"));
        paths.put(Leukocyte_pre, new TestCase(globalPath + "leukocyte-preprocessing/", "leukocyte_preprocessing"));
        paths.put(Leukocyte, new TestCase(globalPath + "leukocyte/", "leukocyte"));
        paths.put(Myocyte, new TestCase(globalPath + "myocyte/", "myocyte"));
        paths.put(Neighbor_pre, new TestCase(globalPath + "nn/", "hurricane_gen"));
        paths.put(Neighbor, new TestCase(globalPath + "nn/", "nn"));
        paths.put(Needle, new TestCase(globalPath + "nw/", "needle"));
        paths.put(Particle, new TestCase(globalPath + "particle/", "particle"));
        paths.put(Pathfinder, new TestCase(globalPath + "pathfinder/", "pathfinder"));
        paths.put(Srad, new TestCase(globalPath + "srad/", "srad"));
        paths.put(Stream, new TestCase(globalPath + "stream/", "streamcluster"));
        //paths.put(Parsing, new TestCase(globalPath + "_Longest_Parsingtime/", "Parsing"));
    }


    public static final HashMap<RodiniaBenchmarks, Integer> splitsize;
    static {
        splitsize = new HashMap<>();
        splitsize.put(Backpropagation, 8192);
        splitsize.put(Breadth_First_Search, 65536);
        splitsize.put(CFD, 24000);
        splitsize.put(Heartwall, 24);
        splitsize.put(Hotspot, 72);
        splitsize.put(Hotspot3D, 24);
        splitsize.put(Kmeans, 65536);
        splitsize.put(LavaMD, 24);
        splitsize.put(Lud, 24);
        splitsize.put(Leukocyte_pre, 8192);
        splitsize.put(Leukocyte, 8192);
        splitsize.put(Myocyte, 24);
        splitsize.put(Neighbor_pre, 240);
        splitsize.put(Neighbor, 240);
        splitsize.put(Needle,96);
        splitsize.put(Particle,1024);
        splitsize.put(Pathfinder, 8192);
        splitsize.put(Srad, 120);
        splitsize.put(Stream, 8192);
        //paths.put(Parsing, new TestCase(globalPath + "_Longest_Parsingtime/", "Parsing"));
    }

}
