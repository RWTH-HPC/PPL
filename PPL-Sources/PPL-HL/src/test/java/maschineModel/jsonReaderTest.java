package maschineModel;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.maschineModel.ClusterDescription;
import de.se_rwth.commons.logging.Log;
import de.parallelpatterndsl.patterndsl.maschineModel.hardwareDescription.Cluster;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public class jsonReaderTest {

    public static final String MODEL_SOURCE_PATH = "./src/test/resources/";

    public static final String VALID = "Valid/";

    public static final String INVALID = "Invalid/";

    @BeforeAll
    public static void disableFailQuick() {
        Log.enableFailQuick(false);
    }

    @ParameterizedTest
    @CsvSource({
            "reader.json",
            "readerWithTemplates.json"
    })
    public void testValid(String StringPath) {
        Cluster cluster = ClusterDescription.parse(MODEL_SOURCE_PATH + VALID + StringPath);
        System.out.println("Cluster latency: " );
        for (Node node1: cluster.getNodes()) {
            for (Node node2: cluster.getNodes()) {
                System.out.println("  " + node1.getIdentifier() + "-" + node2.getIdentifier() + ":" + cluster.getConnectivityLatency(node1.getIdentifier(),node2.getIdentifier()));
            }
        }

        System.out.println(" " );
        System.out.println(" " );

        for (Node node : cluster.getNodes() ) {
            System.out.println("  Node latency: " + node.getIdentifier());

            for (Device device1: node.getDevices() ) {
                System.out.println("    Device latency: " + device1.getIdentifier());
                for (Device device2: node.getDevices() ) {
                    System.out.println("      " + device1.getIdentifier() + "-" + device2.getIdentifier() + ":" + node.getConnectivityLatency(device1.getIdentifier(),device2.getIdentifier()));
                }
                for (Processor processor : device1.getProcessor() ) {
                    System.out.println("      Cache group latency: " + processor.getIdentifier());
                    for (Double x : processor.getCacheLatency()) {
                        System.out.println("        Cache latency: " + x);
                    }
                }
                System.out.println(" " );
            }
            System.out.println(" " );
            System.out.println(" " );
        }
        assertFalse(Log.getErrorCount() > 0);
    }

    @ParameterizedTest
    @CsvSource({
            "parsingError.json",
            "missingParameter.json",
            "missingNodes.json",
            "missingDevices.json",
            "missingCaches.json",
            "missingCacheParameter.json"
    })
    public void testInvalid(String StringPath) {
        try {
            Cluster cluster = ClusterDescription.parse(MODEL_SOURCE_PATH + INVALID + StringPath);

            System.out.println("Cluster latency: " );
            for (Node node1: cluster.getNodes()) {
                for (Node node2: cluster.getNodes()) {
                    System.out.println("  " + node1.getIdentifier() + "-" + node2.getIdentifier() + ":" + cluster.getConnectivityLatency(node1.getIdentifier(),node2.getIdentifier()));
                }
            }

            System.out.println(" " );
            System.out.println(" " );

            for (Node node : cluster.getNodes() ) {
                System.out.println("  Node latency: " + node.getIdentifier());

                for (Device device1: node.getDevices() ) {
                    System.out.println("    Device latency: " + device1.getIdentifier());
                    for (Device device2: node.getDevices() ) {
                        System.out.println("      " + device1.getIdentifier() + "-" + device2.getIdentifier() + ":" + node.getConnectivityLatency(device1.getIdentifier(),device2.getIdentifier()));
                    }
                    for (Processor processor : device1.getProcessor() ) {
                        System.out.println("      Cache group latency: " + processor.getIdentifier());
                        for (Double x : processor.getCacheLatency()) {
                            System.out.println("        Cache latency: " + x);
                        }
                    }
                    System.out.println(" " );
                }
                System.out.println(" " );
                System.out.println(" " );
            }
        } catch (NullPointerException e) {
            e.printStackTrace();
        }


        assertTrue(Log.getErrorCount() > 0);
    }
}