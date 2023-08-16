package Generator;

import Tooling.Tool;
import org.junit.Test;

public class StencilDependencyTest {

    public static final String BENCHMARK_PATH = "./src/test/resources/Generator/stencilDependencyTest.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "1000", "-d", "1000", "-b", "-JSON"};
        Tool.main(args);

    }
}
