package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTLiteralExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if the return name is defined.
 */
public class NoExitInPatternCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (!node.getPatternType().isPresentRecursion() && !node.getPatternType().isPresentSerial()) {
            Helper helper = new Helper();
            if (!helper.hasCorrectAccessPattern(node)) {
                Log.error(node.get_SourcePositionStart() + " Exit used in parallel pattern: " + node.getName());
            }
        }
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
        public void visit(ASTNameExpression node) {
            if (node.getName().equals("exit")) {
                isCorrect = false;
                Log.error(node.get_SourcePositionStart() + " Illegal exit call within parallel pattern!");
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
