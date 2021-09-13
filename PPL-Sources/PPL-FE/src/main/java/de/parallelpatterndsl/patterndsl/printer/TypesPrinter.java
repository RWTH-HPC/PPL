package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl._ast.*;

/**
 * Class that implements a types printer for the PatternDSL
 */
public class TypesPrinter extends AbstractTypesPrinter<String> {
    private static TypesPrinter instance;

    private TypesPrinter() {
    }

    private static TypesPrinter getInstance() {
        if (instance == null) {
            instance = new TypesPrinter();
        }
        return instance;
    }

    public static String printType(ASTType type) {
        return getInstance().doPrintType(type);
    }

    @Override
    protected String doPrintListType(ASTListType type) {
        return "[" + this.doPrintType(type.getType()) + "]";
    }

    @Override
    protected String doPrintNameType(ASTTypeName type) {
        return type.getName();
    }

}
