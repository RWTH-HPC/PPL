package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTModuleCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReturnStatmentHelper;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class LoopSkipOnlyWithinLoopCoCo implements PatternDSLASTModuleCoCo {

    @Override
    public void check(ASTModule node) {
        Helper helper = new Helper();

        node.accept(helper.getRealThis());

    }

    private class Helper implements PatternDSLVisitor {
        private boolean isCorrect = true;

        public Helper() {
        }

        public boolean hasCorrectAccessPattern(ASTFunction node) {
            node.accept(getRealThis());
            return isCorrect;
        }

        @Override
        public void visit(ASTLoopSkipStatement node) {
            Log.error(node.get_SourcePositionStart() + " Break/Continue used outside of Loop!");
        }

        @Override
        public void traverse(ASTForStatement node) {}

        @Override
        public void traverse(ASTWhileStatement node) {}

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
