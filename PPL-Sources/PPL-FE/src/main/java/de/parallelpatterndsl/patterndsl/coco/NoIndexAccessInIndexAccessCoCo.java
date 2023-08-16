package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;

import de.parallelpatterndsl.patterndsl._ast.ASTExpression;

import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTIndexAccessExpressionCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks if the access notation ([]) is used in another access.
 */
public class NoIndexAccessInIndexAccessCoCo implements PatternDSLASTIndexAccessExpressionCoCo {



    @Override
    public void check(ASTIndexAccessExpression node) {
        Helper helper = new Helper();
        if (helper.testForCalls(node.getIndex())) {
            Log.error(node.get_SourcePositionStart() + " Nested Index access expressions detected! ");
            Log.error("This error for nested index accesses is only temporary and nested accesses will be re-enabled in a future release.");
        }

    }

    private class Helper implements PatternDSLVisitor {
        private boolean hasIndexAccess = false;
        private boolean isTesting = true;

        @Override
        public void visit(ASTIndexAccessExpression node) {
            if (isTesting) {
                hasIndexAccess = true;
                Log.error(node.get_SourcePositionStart() + " Do not use index access expressions array[] within other index accesses.");
            }
        }

        public boolean testForCalls(ASTExpression node) {
            node.accept(getRealThis());
            return hasIndexAccess;
        }
    }
}
