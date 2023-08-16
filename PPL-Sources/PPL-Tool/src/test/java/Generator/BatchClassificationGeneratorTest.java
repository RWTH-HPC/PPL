package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class BatchClassificationGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/classification/ppl/batch_classification.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "65536", "-d", "65536", "-b", "-opt", "-o", "out/batch.cxx", "-APT", "-JSON"};
        Tool.main(args);

    }

}
