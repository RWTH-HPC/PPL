package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class NeuralNetworkGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/nn/ppl/neural_network.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "4096", "-d", "4096", "-b", "-JSON", "-o", "out/nn.cxx"};
        Tool.main(args);

    }
}
