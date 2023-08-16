package de.parallelpatterndsl.patterndsl._symboltable;

import de.parallelpatterndsl.patterndsl._ast.ASTPatternType;

/**
 * A symbol for Function, that sores its name and pattern type.
 */
public class FunctionSymbol extends FunctionSymbolTOP {

    /**
     * The pattern implemented by this function.
     */
    ASTPatternType pattern;

    /**
     * The number of parameters the function accepts
     */
    int parameterCount;

    public FunctionSymbol(String name) {
        super(name);
    }

    public FunctionSymbol(String name, ASTPatternType pattern, int parameterCount) {
        super(name);
        this.pattern = pattern;
        this.parameterCount = parameterCount;
    }

    public int getParameterCount() {
        return parameterCount;
    }

    public ASTPatternType getPattern() {
        return this.pattern;
    }
}
