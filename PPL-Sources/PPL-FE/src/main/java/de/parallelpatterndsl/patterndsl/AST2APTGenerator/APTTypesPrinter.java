package de.parallelpatterndsl.patterndsl.AST2APTGenerator;

import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._ast.ASTType;
import de.parallelpatterndsl.patterndsl._ast.ASTTypeName;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.printer.AbstractTypesPrinter;
import de.se_rwth.commons.logging.Log;

import java.util.HashMap;

/**
 * Class that translates the PPL data types in corresponding PrimitiveDataTypes for the APT.
 */
public class APTTypesPrinter extends AbstractTypesPrinter<PrimitiveDataTypes> {

    private static APTTypesPrinter instance;

    /**
     * Hash map containing the dependency between the APT and AST types.
     */
    private HashMap<String,PrimitiveDataTypes> APTTypesMap = new HashMap<>();

    public APTTypesPrinter(){
        APTTypesMap.put("Double", PrimitiveDataTypes.DOUBLE);
        APTTypesMap.put("Float", PrimitiveDataTypes.FLOAT);
        APTTypesMap.put("Int", PrimitiveDataTypes.INTEGER_32BIT);
        APTTypesMap.put("Char", PrimitiveDataTypes.CHARACTER);
        APTTypesMap.put("String", PrimitiveDataTypes.STRING);
        APTTypesMap.put("Bool", PrimitiveDataTypes.BOOLEAN);
        APTTypesMap.put("Int8", PrimitiveDataTypes.INTEGER_8BIT);
        APTTypesMap.put("Int16", PrimitiveDataTypes.INTEGER_16BIT);
        APTTypesMap.put("Int32", PrimitiveDataTypes.INTEGER_32BIT);
        APTTypesMap.put("Int64", PrimitiveDataTypes.INTEGER_64BIT);
        APTTypesMap.put("UInt8", PrimitiveDataTypes.UNSIGNED_INTEGER_8BIT);
        APTTypesMap.put("UInt16", PrimitiveDataTypes.UNSIGNED_INTEGER_16BIT);
        APTTypesMap.put("UInt32", PrimitiveDataTypes.UNSIGNED_INTEGER_32BIT);
        APTTypesMap.put("UInt64", PrimitiveDataTypes.UNSIGNED_INTEGER_64BIT);
        APTTypesMap.put("Long", PrimitiveDataTypes.INTEGER_64BIT);
        APTTypesMap.put("Void", PrimitiveDataTypes.VOID);
    }

    private static APTTypesPrinter getInstance() {
        if (instance == null) {
            instance = new APTTypesPrinter();
        }
        return instance;
    }

    /**
     * Tranforms the given ASTType into its corresponding APT counterpart.
     * @param type
     * @return
     */
    public static PrimitiveDataTypes printType(ASTType type) {
        return getInstance().doPrintType(type);
    }
    
    @Override
    protected PrimitiveDataTypes doPrintListType(ASTListType type) {
        return printType(type.getType());
    }

    @Override
    protected PrimitiveDataTypes doPrintNameType(ASTTypeName type) {
        PrimitiveDataTypes result;
        if (APTTypesMap.containsKey(type.getName())){
            result = APTTypesMap.get(type.getName());
        } else {
            Log.error("Data type \"" + type.getName() + "\" not supported! At line: " + type.get_SourcePositionStart());
            throw new RuntimeException("Critical error!");
        }
        return result;
    }
}
