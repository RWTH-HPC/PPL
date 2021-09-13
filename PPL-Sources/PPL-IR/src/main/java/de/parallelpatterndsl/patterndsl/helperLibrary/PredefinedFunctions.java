package de.parallelpatterndsl.patterndsl.helperLibrary;

import java.util.ArrayList;

public class PredefinedFunctions {

    private static ArrayList<String> functionNames;

    private static boolean isInitialized = false;

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
        }
    }
}
