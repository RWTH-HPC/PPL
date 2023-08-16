package Generator;

import Tooling.Tool;
import org.junit.jupiter.api.Test;

public class JacobiLoopGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/jacobi/ppl/jacobi_loop.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    @Test
    public void test() {

        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "4096", "-d", "4096", "-b", "-opt", "-o", "out/Jacobi.cxx"};
        Tool.main(args);

    }
}
