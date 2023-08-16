package de.parallelpatterndsl.patterndsl._symboltable;
/**
import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
 **/
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTLiteralExpression;

import java.util.Optional;

/**
 * A symbol for constants that stores its name and its optional types.
 * exactly one of the Optional Variables is always non-empty.
 */
public class ConstantSymbol extends ConstantSymbolTOP {

    private Optional<ASTListExpression> listValue;

    private Optional<ASTNameExpression> nameValue;

    private Optional<ASTLiteralExpression> literalValue;

    public ConstantSymbol(String name) {
        super(name);
    }

    public ConstantSymbol(String name, Optional<ASTListExpression> listValue, Optional<ASTNameExpression> nameValue, Optional<ASTLiteralExpression> literalValue) {
        super(name);
        this.listValue = listValue;
        this.nameValue = nameValue;
        this.literalValue = literalValue;
    }

    public Optional<ASTListExpression> getListValue() {
        return listValue;
    }

    public Optional<ASTNameExpression> getNameValue() {
        return nameValue;
    }

    public Optional<ASTLiteralExpression> getLiteralValue() {
        return literalValue;
    }
}
