package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTArgumentsCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTPatternCallStatementCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class NoGlobalsAsArgumentCoCo implements PatternDSLASTPatternCallStatementCoCo {


    @Override
    public void check(ASTPatternCallStatement node) {
        Helper helper = new Helper();
        helper.hasGlobalValue(node);
    }


    private class Helper implements PatternDSLVisitor {
        private boolean isCorrect = true;

        public Helper() {
        }

        public boolean hasGlobalValue(ASTPatternCallStatement node) {
            node.accept(getRealThis());
            return isCorrect;
        }

        @Override
        public void visit(ASTNameExpression node) {
            if (node.getEnclosingScope().resolve(node.getName(), VariableSymbol.KIND).isPresent()) {
                VariableSymbol variableSymbol = (VariableSymbol) node.getEnclosingScope().resolve(node.getName(), VariableSymbol.KIND).get();
                if (variableSymbol.isGlobal()) {
                    Log.error(node.get_SourcePositionStart() + " Global variable " + node.getName() + " used as parallel function argument");
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
