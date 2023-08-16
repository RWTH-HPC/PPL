package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTAssignmentExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTDecrementExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIncrementExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTVariable;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTAssignmentExpressionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTVariableCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class IncrementOnlyAssignmentVariableCoCo implements PatternDSLASTVariableCoCo {


    @Override
    public void check(ASTVariable node) {
        FindUpdateClause tester = new FindUpdateClause();
        if (tester.hasUpdateClause(node)) {
            Log.error(node.get_SourcePositionStart() + "Increment or Decrement (++, --) used in an assignment. Replace with \"+1\" or \"-1\".");
        }
    }

    private class FindUpdateClause implements PatternDSLVisitor {

        private boolean result = false;

        public boolean hasUpdateClause(ASTVariable node){
            node.accept(this.getRealThis());
            return result;
        }

        @Override
        public void visit(ASTIncrementExpression node) {
            result = true;
        }

        @Override
        public void visit(ASTDecrementExpression node) {
            result = true;
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
