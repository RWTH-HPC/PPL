package Generator;

import Tooling.Tool;
import org.junit.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

public class RodiniaGeneratorTest {

    public static final String CLUSTER_SPEC_PATH = "../../Samples/rodinia-ppl/clusters/cluster_c18g.json";

    @ParameterizedTest
    @EnumSource(RodiniaBenchmarks.class)
    public void test(RodiniaBenchmarks rodiniaBenchmarks) {

        TestCase test = RodiniaBenchmarks.paths.get(rodiniaBenchmarks);

        String benchmarkPath = test.getPath() + test.getName() + ".par";

        String outputPath = "../" + test.getPath() + "out/" + test.getName() + ".cpp";

        String[] args = new String[]{"-i", benchmarkPath, "-n", CLUSTER_SPEC_PATH, "-o", outputPath, "-s", "" + RodiniaBenchmarks.splitsize.get(rodiniaBenchmarks), "-d", "" + RodiniaBenchmarks.splitsize.get(rodiniaBenchmarks)};
        Tool.main(args);

    }

}
