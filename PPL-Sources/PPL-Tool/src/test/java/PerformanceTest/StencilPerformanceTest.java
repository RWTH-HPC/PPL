package PerformanceTest;

import Tooling.Tool;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

public class StencilPerformanceTest {

    public static final String BENCHMARK_PATH = "../../Samples/parserPerformance/stencil_3.par";

    public static final String CLUSTER_SPEC_PATH = "../../Samples/clusters/cluster_c18g.json";

    public static final String test = "target/a";
    public static final String Statistics ="../../Samples/parserPerformance/";
    @ParameterizedTest
    @CsvSource({
            "../../Samples/parserPerformance/stencil_3.par",
            "../../Samples/parserPerformance/stencil_3.par",
            "../../Samples/parserPerformance/stencil_6.par",
            "../../Samples/parserPerformance/stencil_9.par",
            "../../Samples/parserPerformance/stencil_12.par",
            "../../Samples/parserPerformance/stencil_15.par",
            "../../Samples/parserPerformance/stencil_18.par",
            "../../Samples/parserPerformance/stencil_21.par",
            "../../Samples/parserPerformance/stencil_24.par",
            "../../Samples/parserPerformance/stencil_27.par",
            "../../Samples/parserPerformance/stencil_30.par"
    })

    public void test(String modelStringPath) {


        String[] args = new String[]{"-i", modelStringPath, "-n", CLUSTER_SPEC_PATH, "-dur"};
        Tool.main(args);


    }
}
