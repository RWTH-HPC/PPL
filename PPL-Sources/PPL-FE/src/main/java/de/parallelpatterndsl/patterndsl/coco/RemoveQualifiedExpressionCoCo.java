package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTConditionalExpression;
import de.monticore.expressions.commonexpressions._ast.ASTQualifiedNameExpression;
import de.monticore.expressions.commonexpressions._cocos.CommonExpressionsASTConditionalExpressionCoCo;
import de.monticore.expressions.commonexpressions._cocos.CommonExpressionsASTQualifiedNameExpressionCoCo;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, Qualified name expressions are used.
 */
public class RemoveQualifiedExpressionCoCo implements CommonExpressionsASTQualifiedNameExpressionCoCo {

    @Override
    public void check(ASTQualifiedNameExpression ast) {
        Log.error(ast.get_SourcePositionStart() + " Qualified name expressions are not supported!" + ast.getName());
    }
}
