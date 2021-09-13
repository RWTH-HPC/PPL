package Generator;

import java.util.HashMap;

/**
 * Enum describing all test cases for a parameterized test.
 * The Cases are encoded in the following way:
 */
public enum RodiniaBenchmarks {

    Backpropagation,
    Breadth_First_Search,
    CFD,
    Heartwall,
    Hotspot,
    Hotspot3D,
    Kmeans,
    LavaMD,
    Lud,
    Leukocyte_pre,
    Leukocyte,
    Myocyte,
    Neighbor_pre,
    Neighbor,
    Needle,
    Particle,
    Pathfinder,
    Srad,
    Stream,
    //Parsing
    ;


    private static final String globalPath = "../../Samples/Rodinia-PPL/";

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
        paths.put(Lud, new TestCase(globalPath + "Lud/", "lud"));
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


}
