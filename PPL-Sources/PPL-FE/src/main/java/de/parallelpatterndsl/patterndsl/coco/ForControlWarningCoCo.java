package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTExpression;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTCommonForControlCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTForControlCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTForEachControlCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * Defines a CoCo that checks, that the variable in for loops is not assigned.
 */
public class ForControlWarningCoCo implements PatternDSLASTCommonForControlCoCo {

    @Override
    public void check(ASTCommonForControl node) {
        String controlVar = "";

        // compute the loop control variable
        if (node.getForInit().getVariableOpt().isPresent()) {
            controlVar = node.getForInit().getVariable().getName();
        } else if (node.getForInit().getExpressionOpt().isPresent()) {
            if (node.getExpression() instanceof ASTAssignmentExpression) {
                ASTAssignmentExpression expression = (ASTAssignmentExpression) node.getExpression();
                if (expression.getLeft() instanceof ASTNameExpression) {
                    controlVar = ((ASTNameExpression) expression.getLeft()).getName();
                } else if(expression.getLeft() instanceof ASTIndexAccessExpression) {
                    ASTExpression runningExpression = expression.getLeft();
                    while (runningExpression instanceof ASTIndexAccessExpression) {
                        runningExpression = ((ASTIndexAccessExpression) runningExpression).getIndexAccess();
                    }
                    ASTNameExpression astNameExpression = (ASTNameExpression) runningExpression;
                    controlVar = astNameExpression.getName();
                }else {
                    Log.error(node.get_SourcePositionStart() + " Loop control is ill defined!");
                }

            } else {
                Log.error(node.get_SourcePositionStart() + " No assignment for loop control variable!");
            }
        } else {
            Log.error(node.get_SourcePositionStart() + " No assignment for loop control variable!");
        }

        ControlTraversal helper = new ControlTraversal(controlVar);

        if (!helper.find(node.getCondition())) {
            Log.warn(node.getCondition().get_SourcePositionStart() + " Loop control variable " + controlVar + " is not used in loop condition!");
        }

        if (!helper.find(node.getExpression())) {
            Log.warn(node.getExpression().get_SourcePositionStart() + " Loop control variable " + controlVar + " is not used in loop update expression!");
        }


    }

    /**
     * Support class to find uses of the control variable.
     */
    private class ControlTraversal implements PatternDSLVisitor {

        private String toFind;

        private boolean found = false;

        public ControlTraversal(String toFind) {
            this.toFind = toFind;
        }

        public boolean find(ASTExpression expression) {
            found = false;
            expression.accept(this.getRealThis());
            return found;
        }

        @Override
        public void visit(ASTNameExpression node) {
            if (node.getName().equals(toFind)) {
                found = true;
            }
        }

        private PatternDSLVisitor realThis = this;

        @Override
        public PatternDSLVisitor getRealThis() {
            return realThis;
        }

        @Override
        public void setRealThis(PatternDSLVisitor realThis) {
            this.realThis = realThis;
        }
    }

}
