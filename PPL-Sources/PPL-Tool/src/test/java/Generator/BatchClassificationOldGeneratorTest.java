package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class BatchClassificationOldGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/classification/ppl/batch_classification_loop.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "262144", "-d", "262144", "-b", "-opt", "-o", "out/batch.cxx"};
        Tool.main(args);

    }

}
