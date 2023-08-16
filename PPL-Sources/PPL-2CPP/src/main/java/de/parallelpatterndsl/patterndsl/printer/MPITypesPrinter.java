package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;

import java.util.HashMap;

/**
 * Prints the Primitive data types for c++.
 */
public class MPITypesPrinter {

    private static MPITypesPrinter instance;

    private HashMap<PrimitiveDataTypes,String> CppTypesMap = new HashMap<>();

    public MPITypesPrinter(){
        CppTypesMap.put(PrimitiveDataTypes.BOOLEAN, "MPI_BOOL");
        CppTypesMap.put(PrimitiveDataTypes.CHARACTER, "MPI_CHAR");
        CppTypesMap.put(PrimitiveDataTypes.DOUBLE, "MPI_DOUBLE");
        CppTypesMap.put(PrimitiveDataTypes.FLOAT, "MPI_FLOAT");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_16BIT, "MPI_SHORT");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_32BIT, "MPI_INT");
        CppTypesMap.put(PrimitiveDataTypes.INTEGER_64BIT, "MPI_LONG");
        CppTypesMap.put(PrimitiveDataTypes.STRING, "MPI_CHAR");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_16BIT, "MPI_UNSIGNED_SHORT");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_32BIT, "MPI_UNSIGNED_INT");
        CppTypesMap.put(PrimitiveDataTypes.UNSIGNED_INTEGER_64BIT, "MPI_UNSIGNED_LONG");
    }

    public static String doPrintType(PrimitiveDataTypes type) {
        if (instance == null) {
            instance = new MPITypesPrinter();
        }
        return instance.printType(type);
    }

    private String printType(PrimitiveDataTypes type) {
        if (!CppTypesMap.containsKey(type)) {
            throw new RuntimeException("type not printable in MPI!" + type.name());
        }
        return CppTypesMap.get(type);
    }


}
