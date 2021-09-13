package de.parallelpatterndsl.patterndsl.MappingTree;

import java.util.HashSet;

public class SupportFunction {

    public static HashSet getElementSet(Object obj) {
        HashSet res = new HashSet();
        res.add(obj);
        return res;
    }

    public static long min(long first, long second) {
        if (first < second) {
            return first;
        }
        return second;
    }

    public static long max(long first, long second) {
        if (first > second) {
            return first;
        }
        return second;
    }
}
