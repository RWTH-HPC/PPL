package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTReturnStatementCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReturnStatmentHelper;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class ReturnWithoutCallCoCo implements PatternDSLASTReturnStatementCoCo {

    @Override
    public void check(ASTReturnStatement node) {
        Helper helper = new Helper();
        helper.hasCorrectAccessPattern(node);
    }
    private class Helper implements PatternDSLVisitor {
        private boolean isCorrect = true;

        public Helper() {
        }

        public boolean hasCorrectAccessPattern(ASTReturnStatement node) {
            node.accept(getRealThis());
            return isCorrect;
        }

        @Override
        public void visit(ASTCallExpression node) {
            Log.error("Call inside Return not allowed", node.get_SourcePositionStart());
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
