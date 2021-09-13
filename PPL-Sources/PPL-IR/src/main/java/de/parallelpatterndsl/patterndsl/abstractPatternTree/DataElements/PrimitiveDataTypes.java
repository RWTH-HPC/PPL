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



    public static int GetPrimitiveSize(PrimitiveDataTypes types) {
        return SizeOfPrimitiveDataTypes.get(types);
    }

}
