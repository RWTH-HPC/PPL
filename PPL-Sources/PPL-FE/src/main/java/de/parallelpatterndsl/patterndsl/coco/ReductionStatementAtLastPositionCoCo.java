package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTCallExpression;
import de.monticore.expressions.commonexpressions._ast.ASTSimpleAssignmentExpression;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if the last action of a reduction is a reductionStatement.
 */
public class ReductionStatementAtLastPositionCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (!node.getPatternType().isPresentReduction()) {
            return;
        }
        ASTBlockElement last = node.getBlockStatement().getBlockElement(node.getBlockStatement().sizeBlockElements() - 1 );
        if (last.isPresentExpression()) {
            ASTExpression exp = last.getExpression();
            if (exp instanceof ASTAssignmentByMultiplyExpression || exp instanceof ASTAssignmentByDecreaseExpression || exp instanceof ASTAssignmentByIncreaseExpression) {
                return;
            } else if (exp instanceof ASTAssignmentExpression) {
                exp = ((ASTAssignmentExpression) exp).getRight();
                if (exp instanceof ASTCallExpression) {
                    return;
                }
            } else if(exp instanceof ASTSimpleAssignmentExpression) {
                if(((ASTSimpleAssignmentExpression) exp).getOperator().equals("+=")) {
                    return;
                }
            }
        }
        Log.error(node.get_SourcePositionStart() + " No reduction statement found in: " + node.getName());

    }



}
