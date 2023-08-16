package CMD;

import java.util.HashMap;
import java.util.Map;

/**
 * Enum describing all test cases for a parameterized test.
 * The Cases are encoded in the following way:
 */
public enum ToolOptions {

    HELP,
    VERSION,
    INPUTPATH,
    OPTREPORT,
    OUTPUTPATH,
    APT,
    CALL,
    FULL,
    LOOKAHEAD,
    OVERLAP,
    BASEMAPPING,
    SPLITSIZE,
    DATASPLITSIZE,
    CLUSTERPATH,
    RANDNAME,
    TIMELIMIT,
    JSON,
    INFO,
    DURATION,
    MEMORY,
    INLINE,
    DEFAULTDEVICE,
    DEFAULTNODE,
    EXPLICITMAPPING,
    UNROLL,
    GPUTHREAD,
    ;

    public static final HashMap<ToolOptions, Option> options;
    public static final HashMap<String, ToolOptions> Flags;
    static {
        options = new HashMap<>();
        options.put(HELP,new Option<>(0, "Get a description of the tool", false, "-h", "--help", false, Integer.class));
        options.put(VERSION,new Option<>(0, "Get the version of the tool", false, "-v", "--version", false, Integer.class));
        options.put(INPUTPATH,new Option<>("", "The target file to be optimized. Target files may not have a '-' in their name", true, "-i", "--input", true, String.class));
        options.put(OPTREPORT,new Option<>(false, "Print the optimization report", false, "-opt", "--optreport", false, Boolean.class));
        options.put(OUTPUTPATH,new Option<>("a.cpp", "The output path for the generated code", false, "-o", "--output", true, String.class));
        options.put(APT,new Option<>(false, "Print the theoretical APT", false, "-APT", "--printTheoreticalAPT", false, Boolean.class));
        options.put(CALL,new Option<>(false, "Print the Call Graph", false, "-CALL", "--printCallGraph", false, Boolean.class));
        options.put(FULL,new Option<>(false, "Print the fully nested APT", false, "-FULL", "--printFullAPT", false, Boolean.class));
        options.put(LOOKAHEAD,new Option<>(1, "Define the lookahead for the mapping optimization", false, "-l", "--lookahead", true, Integer.class));
        options.put(OVERLAP,new Option<>(0.0, "Define the overlap of network cost and execution cost", false, "-over", "--overlap", true, Double.class));
        options.put(BASEMAPPING,new Option<>(false, "Print and generate a default mapping", false, "-b", "--baseMapping", false, Boolean.class));
        options.put(SPLITSIZE,new Option<>(128, "The size in which the index range will be grouped", false, "-s", "--splitSize", true, Integer.class));
        options.put(DATASPLITSIZE,new Option<>(1024, "The size in which the data elements will be grouped", false, "-d", "--dataSplitSize", true, Integer.class));
        options.put(CLUSTERPATH,new Option<>("", "The path to the network definition", true, "-n", "--network", true, String.class));
        options.put(RANDNAME, new Option<>(10, "The length of the random name extension", false, "-rand", "--randomNameExtension", true, Integer.class));
        options.put(TIMELIMIT, new Option<>(240, "The maximum duration for a single ILP to be solved", false, "-t", "--timeLimitILP", true, Integer.class));
        options.put(JSON, new Option<>(false, "Print the APT in the JSON Format", false, "-JSON", "--APT2JSON", false, Boolean.class));
        options.put(INFO, new Option<>(1, "Print different levels of compiling information. 0 = only Warning and errors, 1 = info messages, 2 = debug messages", false, "-info", "--InfoLevel", true, Integer.class));
        options.put(DURATION, new Option<>(false, "Store the duration of different steps of compilation in a separate file.", false, "-dur", "--durationPerStep", false, Boolean.class));
        options.put(MEMORY, new Option<>(false, "Store the memory usage of different steps in a separate file.", false, "-mem", "--memoryUsage", false, Boolean.class));
        options.put(INLINE, new Option<>(false, "Set this Flag to avoid inlining of Scopes/functions to increase potential parallelism.", false, "-noin", "--noInlining", false, Boolean.class));
        options.put(DEFAULTDEVICE, new Option<>(0, "Set the device with the given id to the default device on the default node.", false, "-ddiv", "--defaultDevice", true, Integer.class));
        options.put(DEFAULTNODE, new Option<>(0, "Set the node with the given id to the default node calculating sequential code.", false, "-dnode", "--defaultNode", true, Integer.class));
        options.put(EXPLICITMAPPING, new Option<>(false, "Turn off the global mapping optimization, to use predefined splits within the source code.", false, "-exm", "--explicitMode", false, Boolean.class));
        options.put(UNROLL, new Option<>(false, "Turn off the unrolling of simple for loops before the optimization.", false, "-noro", "--NoUnrolling", false, Boolean.class));
        options.put(GPUTHREAD, new Option<>(0, "Pinns the GPU management thread to this thread.", false, "-gpup", "--gpupinthread", true, Integer.class));
        Flags = new HashMap<>();
        generateFlags();
    }

    public static void getHelpMessage() {
        String output = "TODO: Help";
        StringBuilder call = new StringBuilder();
        StringBuilder definitions = new StringBuilder();
        call.append("java --add-opens java.base/java.lang=ALL-UNNAMED -jar .\\\"nameOfYourTool\".jar ");

        for (Option option: options.values() ) {
            if (!option.isRequired()) {
                call.append("[");
            }
            call.append(option.getShortFlag());
            if (!option.isRequired()) {
                call.append("] ");
            } else {
                call.append(" ");
            }

            definitions.append("\t");
            definitions.append(option.getShortFlag());
            definitions.append(",\t");
            definitions.append(option.getLongFlag());
            if (option.isHasValue()) {
                definitions.append("=<value>");
            }
            definitions.append("\t\t");
            definitions.append(option.getDescription());
            definitions.append(".");
            if (option.isHasValue()) {
                definitions.append(" Default value: ");
                definitions.append(option.getDefaultValue());
            }
            definitions.append("\n");
        }

        call.append("\n");
        call.append(definitions);

        System.out.println(call.toString());
        System.exit(0);
    }

    public static void getVersion() {
        String output = "Version 0.0.1";
        System.out.println(output);
        System.exit(0);
    }

    private static void generateFlags() {
        for (Map.Entry x : options.entrySet()) {
            Flags.put(((Option) x.getValue()).getShortFlag(), (ToolOptions) x.getKey());
            Flags.put(((Option) x.getValue()).getLongFlag(), (ToolOptions) x.getKey());
        }
    }

}
