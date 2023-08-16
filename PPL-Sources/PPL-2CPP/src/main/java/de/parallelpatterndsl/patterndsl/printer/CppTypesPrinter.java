package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;

import java.util.HashMap;

/**
 * Prints the Primitive data types for c++.
 */
public class CppTypesPrinter{

    private static CppTypesPrinter instance;

    private HashMap<PrimitiveDataTypes,String> CppTypesMap = new HashMap<>();

    public CppTypesPrinter(){
        CppTypesMap.put(PrimitiveDataTypes.BOOLEAN, "bool");
        CppTypesMap.put(PrimitiveDataTypes.CHARACTER, "char");
        CppTypesMap.put(PrimitiveDataTypes.DOUBLE, "double");
        CppTypesMap.put(PrimitiveDataTypes.FLOAT, "float");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_8BIT, "int8_t");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_16BIT, "int16_t");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_32BIT, "int32_t");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_64BIT, "int64_t");
        CppTypesMap.put(PrimitiveDataTypes.STRING, "std::string");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_8BIT, "uint8_t");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_16BIT, "uint16_t");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_32BIT, "uint32_t");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_64BIT, "uint64_t");
        CppTypesMap.put(PrimitiveDataTypes.VOID, "void");
    }

    public static String doPrintType(PrimitiveDataTypes type) {
        if (instance == null) {
            instance = new CppTypesPrinter();
        }
        return instance.printType(type);
    }

    private String printType(PrimitiveDataTypes type) {
        if (type == PrimitiveDataTypes.COMPLEX_TYPE) {
            throw new RuntimeException("type not printable! " + type.name());
        }
        return CppTypesMap.get(type);
    }


}
