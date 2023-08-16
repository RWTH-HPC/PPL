package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

import java.util.HashMap;

/**
 * enum defining all PrimitiveDataTypes.
 */
public enum PrimitiveDataTypes {
    VOID,
    BOOLEAN,
    CHARACTER,
    STRING,
    INTEGER_8BIT,
    INTEGER_16BIT,
    INTEGER_32BIT,
    INTEGER_64BIT,
    UNSIGNED_INTEGER_8BIT,
    UNSIGNED_INTEGER_16BIT,
    UNSIGNED_INTEGER_32BIT,
    UNSIGNED_INTEGER_64BIT,
    FLOAT,
    DOUBLE,
    COMPLEX_TYPE;

    private static final HashMap<PrimitiveDataTypes,Integer> SizeOfPrimitiveDataTypes;
    static {
        SizeOfPrimitiveDataTypes = new HashMap<>();
        SizeOfPrimitiveDataTypes.put(VOID,0);
        SizeOfPrimitiveDataTypes.put(BOOLEAN,1);
        SizeOfPrimitiveDataTypes.put(CHARACTER,1);
        SizeOfPrimitiveDataTypes.put(STRING,1);
        SizeOfPrimitiveDataTypes.put(INTEGER_8BIT,1);
        SizeOfPrimitiveDataTypes.put(INTEGER_16BIT,2);
        SizeOfPrimitiveDataTypes.put(INTEGER_32BIT,4);
        SizeOfPrimitiveDataTypes.put(INTEGER_64BIT,8);
        SizeOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_8BIT,1);
        SizeOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_16BIT,2);
        SizeOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_32BIT,4);
        SizeOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_64BIT,8);
        SizeOfPrimitiveDataTypes.put(FLOAT,4);
        SizeOfPrimitiveDataTypes.put(DOUBLE,8);
        SizeOfPrimitiveDataTypes.put(COMPLEX_TYPE,0);
    }

    private static final HashMap<PrimitiveDataTypes,String> StringNames;
    static {
        StringNames = new HashMap<>();
        StringNames.put(VOID,"void");
        StringNames.put(BOOLEAN,"boolean");
        StringNames.put(CHARACTER,"character");
        StringNames.put(STRING,"string");
        StringNames.put(INTEGER_8BIT,"int8");
        StringNames.put(INTEGER_16BIT,"int16");
        StringNames.put(INTEGER_32BIT,"int32");
        StringNames.put(INTEGER_64BIT,"int64");
        StringNames.put(UNSIGNED_INTEGER_8BIT,"uint8");
        StringNames.put(UNSIGNED_INTEGER_16BIT,"uint16");
        StringNames.put(UNSIGNED_INTEGER_32BIT,"uint32");
        StringNames.put(UNSIGNED_INTEGER_64BIT,"uint64");
        StringNames.put(FLOAT,"float");
        StringNames.put(DOUBLE,"double");
        StringNames.put(COMPLEX_TYPE,"complex");
    }

    private static final HashMap<PrimitiveDataTypes,String> MaxOfPrimitiveDataTypes;
    static {
        MaxOfPrimitiveDataTypes = new HashMap<>();
        MaxOfPrimitiveDataTypes.put(VOID,"0");
        MaxOfPrimitiveDataTypes.put(BOOLEAN,"false");
        MaxOfPrimitiveDataTypes.put(CHARACTER,"");
        MaxOfPrimitiveDataTypes.put(STRING,"");
        MaxOfPrimitiveDataTypes.put(INTEGER_8BIT,"127");
        MaxOfPrimitiveDataTypes.put(INTEGER_16BIT,"32767");
        MaxOfPrimitiveDataTypes.put(INTEGER_32BIT,"2147483647");
        MaxOfPrimitiveDataTypes.put(INTEGER_64BIT,"9223372036854775807");
        MaxOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_8BIT,"255");
        MaxOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_16BIT,"65535");
        MaxOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_32BIT,"4294967295");
        MaxOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_64BIT,"18446744073709551615");
        MaxOfPrimitiveDataTypes.put(FLOAT,"3.40282e+038");
        MaxOfPrimitiveDataTypes.put(DOUBLE,"1.79769e+308");
        MaxOfPrimitiveDataTypes.put(COMPLEX_TYPE,"");
    }
    private static final HashMap<PrimitiveDataTypes,String> MinOfPrimitiveDataTypes;
    static {
        MinOfPrimitiveDataTypes = new HashMap<>();
        MinOfPrimitiveDataTypes.put(VOID,"0");
        MinOfPrimitiveDataTypes.put(BOOLEAN,"true");
        MinOfPrimitiveDataTypes.put(CHARACTER,"");
        MinOfPrimitiveDataTypes.put(STRING,"");
        MinOfPrimitiveDataTypes.put(INTEGER_8BIT,"-128");
        MinOfPrimitiveDataTypes.put(INTEGER_16BIT,"-32768");
        MinOfPrimitiveDataTypes.put(INTEGER_32BIT,"-2147483648");
        MinOfPrimitiveDataTypes.put(INTEGER_64BIT,"-9223372036854775808");
        MinOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_8BIT,"0");
        MinOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_16BIT,"0");
        MinOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_32BIT,"0");
        MinOfPrimitiveDataTypes.put(UNSIGNED_INTEGER_64BIT,"0");
        MinOfPrimitiveDataTypes.put(FLOAT,"1.17549e-038");
        MinOfPrimitiveDataTypes.put(DOUBLE,"2.22507e-308");
        MinOfPrimitiveDataTypes.put(COMPLEX_TYPE,"");
    }


    public static String getMinValue(PrimitiveDataTypes types) {
        return MinOfPrimitiveDataTypes.get(types);
    }
    public static String getMaxValue(PrimitiveDataTypes types) {
        return MaxOfPrimitiveDataTypes.get(types);
    }


    public static int GetPrimitiveSize(PrimitiveDataTypes types) {
        return SizeOfPrimitiveDataTypes.get(types);
    }

    public static String toString(PrimitiveDataTypes type) {
        return StringNames.get(type);
    }



}
