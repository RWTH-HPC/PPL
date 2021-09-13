package Generator;

/**
 * Class to describe a test case to simplify junit tests.
 */
public class TestCase {
    /**
     * The path to the test case
     */
    private String path;

    /**
     * The name of the test case
     */
    private String name;

    public TestCase(String path, String name) {
        this.path = path;
        this.name = name;
    }

    public String getPath() {
        return path;
    }

    public String getName() {
        return name;
    }
}
