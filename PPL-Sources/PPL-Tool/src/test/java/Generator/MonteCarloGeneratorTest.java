package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class MonteCarloGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/monte_carlo/ppl/monte_carlo.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "24", "-d", "24", "-b", "-JSON", "-o", "out/Monte.cxx", "-seed" ,"1"};
        Tool.main(args);

    }
}
