package de.parallelpatterndsl.patterndsl._symboltable;

import de.parallelpatterndsl.patterndsl._ast.ASTType;

/**
 * A symbol that stores name and type for function parameters.
 */
public class FunctionParameterSymbol extends FunctionParameterSymbolTOP {

    /**
     * The data type of the parameter.
     */
    private ASTType type;

    public FunctionParameterSymbol(String name) {
        super(name);
    }

    public FunctionParameterSymbol(String name, ASTType type) {
        super(name);
        this.type = type;
    }

    public ASTType getType() { return this.type;}

}
