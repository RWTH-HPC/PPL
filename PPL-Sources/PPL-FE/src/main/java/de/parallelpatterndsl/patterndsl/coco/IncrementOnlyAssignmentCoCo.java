package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.assignmentexpressions._cocos.AssignmentExpressionsASTAssignmentExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTAssignmentExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTDecrementExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIncrementExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTAssignmentExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class IncrementOnlyAssignmentCoCo implements PatternDSLASTAssignmentExpressionCoCo {


    @Override
    public void check(ASTAssignmentExpression node) {
        FindUpdateClause tester = new FindUpdateClause();
        if (tester.hasUpdateClause(node)) {
            Log.error(node.get_SourcePositionStart() + "Increment or Decrement (++, --) used in an assignment. Replace with \"+1\" or \"-1\".");
        }
    }

    private class FindUpdateClause implements PatternDSLVisitor {

        private boolean result = false;

        public boolean hasUpdateClause(ASTAssignmentExpression node){
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
