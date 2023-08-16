package Generator;

import Tooling.Tool;
import org.junit.Test;

import java.io.IOException;

public class LuleshGeneratorTest {

    public static final String BENCHMARK_PATH = "../../Samples/lulesh-ppl/lulesh.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/lulesh-ppl/clusters/cluster_c18_ws_8.json";

    @Test
    public void test() throws IOException {
        String command = "cd ../../Samples/lulesh-ppl/; bash preprocessing.sh";
        ProcessBuilder pb = new ProcessBuilder("bash", "-c", command);
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        System.out.println("Total memory: " + Runtime.getRuntime().maxMemory());
        String[] args = new String[]{"-i", BENCHMARK_PATH, "-n", CLUSTER_SPEC_PATH, "-s", "48", "-d", "48", "-o", "out/lulesh.cxx", "-opt", "-APT", "-CALL"};
        Tool.main(args);

    }

}
