package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._symboltable.ConstantSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;

import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTConditionalExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTConditionalExpression;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, Conditional expressions are used.
 */
public class RemoveConditionalExpressionCoCo implements PatternDSLASTConditionalExpressionCoCo {

    @Override
    public void check(ASTConditionalExpression astConditionalExpression) {
        Log.error(astConditionalExpression.get_SourcePositionStart() + " Conditional expressions are not supported!");
    }
}
