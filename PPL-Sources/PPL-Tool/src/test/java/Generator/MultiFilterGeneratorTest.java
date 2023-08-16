package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class MultiFilterGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/multi-filter/ppl/multi_filter.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "4096", "-d", "128", "-b", "-JSON", "-o", "out/Multifilter.cxx"};
        Tool.main(args);

    }
}
