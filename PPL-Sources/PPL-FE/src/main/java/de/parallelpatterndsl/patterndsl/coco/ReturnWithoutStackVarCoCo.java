package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTReturnStatement;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTReturnStatementCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class ReturnWithoutStackVarCoCo implements PatternDSLASTReturnStatementCoCo {

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
        public void visit(ASTNameExpression node) {
            if (node.getEnclosingScope().resolve(node.getName(), VariableSymbol.KIND).isPresent()) {
                if (((VariableSymbol) node.getEnclosingScope().resolve(node.getName(), VariableSymbol.KIND).get()).isArrayOnStack()) {
                    Log.error("You are not allowed to return local Variables: " + node.getName(), node.get_SourcePositionStart());
                }
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
