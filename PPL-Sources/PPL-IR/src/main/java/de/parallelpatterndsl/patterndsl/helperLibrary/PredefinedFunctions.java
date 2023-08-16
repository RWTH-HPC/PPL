package de.parallelpatterndsl.patterndsl.helperLibrary;

import de.se_rwth.commons.SourcePosition;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;

public class PredefinedFunctions {

    private static ArrayList<String> functionNames;

    private static boolean isInitialized = false;

    private static HashMap<String, Integer> minimalParameter;

    private static HashMap<String, Integer> maximumParameter;

    public static boolean contains(String name) {
        init();
        for (String functionName: functionNames ) {
            if (functionName.equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static void init() {
        if (!isInitialized) {
            isInitialized = true;
            functionNames = new ArrayList<>();
            functionNames.add("copy");
            functionNames.add("init_List");
            functionNames.add("concat");
            functionNames.add("min");
            functionNames.add("max");
            functionNames.add(("isnan"));
            functionNames.add(("isinf"));
            functionNames.add(("rand"));
            functionNames.add(("sqrt"));
            functionNames.add(("cbrt"));
            functionNames.add(("pow"));
            functionNames.add(("get_time"));
            functionNames.add(("get_time_nano"));
            functionNames.add("abs");
            functionNames.add("fabs");
            functionNames.add("exit");
            functionNames.add("tanh");
            functionNames.add("get_maxRusage");
            functionNames.add("exp");
            functionNames.add("sin");
            functionNames.add("cos");
            functionNames.add("log");
            functionNames.add("log10");
            functionNames.add("fmod");
            functionNames.add("Cast2Int8");
            functionNames.add("Cast2Int16");
            functionNames.add("Cast2Int");
            functionNames.add("Cast2Long");
            functionNames.add("Cast2Float");
            functionNames.add("Cast2Double");
            functionNames.add("Cast2String");
        }
    }

    public static int minParameters(String name, SourcePosition position) {
        if (functionNames.contains(name)) {
            return minimalParameter.get(name);
        }
        Log.error(position+ " Testing call of predefined function which is not predefined: " + name);
        System.exit(1);
        return 1;
    }

    public static int maxParameters(String name, SourcePosition position) {
        if (functionNames.contains(name)) {
            return maximumParameter.get(name);
        }
        Log.error(position+ " Testing call of predefined function which is not predefined: " + name);
        System.exit(1);
        return 1;
    }

    static {
        minimalParameter = new HashMap<>();
        minimalParameter.put("copy", 1);
        minimalParameter.put("init_List", 1);
        minimalParameter.put("concat", 2);
        minimalParameter.put("min", 2);
        minimalParameter.put("max", 2);
        minimalParameter.put("isnan", 1);
        minimalParameter.put("isinf", 1);
        minimalParameter.put("rand", 0);
        minimalParameter.put("sqrt", 1);
        minimalParameter.put("cbrt", 1);
        minimalParameter.put("pow", 2);
        minimalParameter.put("get_time", 0);
        minimalParameter.put("get_time_nano", 0);
        minimalParameter.put("abs", 1);
        minimalParameter.put("fabs", 1);
        minimalParameter.put("exit", 1);
        minimalParameter.put("tanh", 1);
        minimalParameter.put("get_maxRusage", 0);
        minimalParameter.put("exp", 1);
        minimalParameter.put("sin", 1);
        minimalParameter.put("cos", 1);
        minimalParameter.put("log", 1);
        minimalParameter.put("log10", 1);
        minimalParameter.put("fmod", 2);
        minimalParameter.put("Cast2Int8", 1);
        minimalParameter.put("Cast2Int16", 1);
        minimalParameter.put("Cast2Int", 1);
        minimalParameter.put("Cast2Long", 1);
        minimalParameter.put("Cast2Float", 1);
        minimalParameter.put("Cast2Double", 1);
        minimalParameter.put("Cast2String", 1);

        maximumParameter = new HashMap<>();
        maximumParameter.put("copy", 1);
        maximumParameter.put("init_List", 2);
        maximumParameter.put("concat", 2);
        maximumParameter.put("min", 2);
        maximumParameter.put("max", 2);
        maximumParameter.put("isnan", 1);
        maximumParameter.put("isinf", 1);
        maximumParameter.put("rand", 1);
        maximumParameter.put("sqrt", 1);
        maximumParameter.put("cbrt", 1);
        maximumParameter.put("pow", 2);
        maximumParameter.put("get_time", 0);
        maximumParameter.put("get_time_nano", 0);
        maximumParameter.put("abs", 1);
        maximumParameter.put("fabs", 1);
        maximumParameter.put("exit", 1);
        maximumParameter.put("tanh", 1);
        maximumParameter.put("get_maxRusage", 0);
        maximumParameter.put("exp", 1);
        maximumParameter.put("sin", 1);
        maximumParameter.put("cos", 1);
        maximumParameter.put("log", 1);
        maximumParameter.put("log10", 1);
        maximumParameter.put("fmod", 2);
        maximumParameter.put("Cast2Int8", 1);
        maximumParameter.put("Cast2Int16", 1);
        maximumParameter.put("Cast2Int", 1);
        maximumParameter.put("Cast2Long", 1);
        maximumParameter.put("Cast2Float", 1);
        maximumParameter.put("Cast2Double", 1);
        maximumParameter.put("Cast2String", 1);
    }
}
