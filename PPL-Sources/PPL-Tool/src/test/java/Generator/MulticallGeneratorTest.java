package Generator;

import Tooling.Tool;
import org.junit.Test;

public class MulticallGeneratorTest {

    public static final String BENCHMARK_PATH = "./src/test/resources/Generator/multicalltest.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "4096", "-d", "4096", "-b", "-opt"};
        Tool.main(args);

    }
}
