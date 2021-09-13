package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTConditionalExpression;
import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.commonexpressions._cocos.CommonExpressionsASTConditionalExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.ConstantSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, Conditional expressions are used.
 */
public class RemoveConditionalExpressionCoCo implements CommonExpressionsASTConditionalExpressionCoCo {

    @Override
    public void check(ASTConditionalExpression astConditionalExpression) {
        Log.error(astConditionalExpression.get_SourcePositionStart() + " Conditional expressions are not supported!");
    }
}
