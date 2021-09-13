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

    public FunctionSymbol(String name) {
        super(name);
    }
    public FunctionSymbol(String name, ASTPatternType pattern) {
        super(name);
        this.pattern = pattern;
    }

    public ASTPatternType getPattern() {
        return this.pattern;
    }
}
