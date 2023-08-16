package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTPatternCallStatementCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTVariableCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks if the access notation ([]) is used in pattern calls.
 */
public class NoIndexAccessInPatternCallsCoCo implements PatternDSLASTFunctionCoCo {



    @Override
    public void check(ASTFunction node) {
        Helper helper = new Helper();
        if (node.getPatternType().isPresentSerial()) {
            if (helper.testForCalls(node)) {
                //TODO: Change back if no further error
               //Log.error(node.get_SourcePositionStart() + " Index access expressions as arguments to pattern call detected: " + node.getName());
            }
        }

    }

    private class Helper implements PatternDSLVisitor {
        private boolean hasIndexAccess = false;
        private boolean isTesting = false;

        @Override
        public void visit(ASTPatternCallStatement node) {
            isTesting = true;
        }

        @Override
        public void endVisit(ASTPatternCallStatement node) {
            isTesting = false;
        }

        @Override
        public void visit(ASTIndexAccessExpression node) {
            if (isTesting) {
                hasIndexAccess = true;
                //TODO: Change if no futher error
                //Log.error(node.get_SourcePositionStart() + " Do not use index access expressions array[] as arguments to pattern calls.");
            }
        }

        public boolean testForCalls(ASTFunction node) {
            node.accept(getRealThis());
            return hasIndexAccess;
        }
    }
}
