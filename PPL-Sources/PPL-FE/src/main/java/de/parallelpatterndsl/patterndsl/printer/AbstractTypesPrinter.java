package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl._ast.*;

/**
 * Abstract types printer. This class defines which functions must be implemented for types printer.
 * @param <T>
 */
public abstract class AbstractTypesPrinter<T> {

    protected AbstractTypesPrinter() {}

    protected T doPrintType(ASTType type) {
        if (type instanceof ASTListType) {
            return this.doPrintListType((ASTListType)type);
        } else if(type instanceof ASTTypeName) {
            return doPrintNameType((ASTTypeName) type);
        }

        throw new RuntimeException("Sub-class of ASTType not implemented");
    }

    protected abstract T doPrintListType(ASTListType type);

    protected abstract T doPrintNameType(ASTTypeName type);
}
