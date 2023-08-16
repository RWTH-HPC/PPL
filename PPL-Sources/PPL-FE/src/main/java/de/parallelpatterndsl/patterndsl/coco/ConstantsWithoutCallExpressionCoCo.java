package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTConstant;
import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTConstantCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * Defines a CoCo that checks whether a constant definition contains a function call.
 */
public class ConstantsWithoutCallExpressionCoCo implements PatternDSLASTConstantCoCo {

    @Override
    public void check(ASTConstant node) {
        Helper helper = new Helper();
        if (!helper.testForCalls(node)) {
            Log.error(node.get_SourcePositionStart() + " Call Expression within constant definition");
        }
    }

    /**
     * Internal helper, that utilizes visitors to traverse the constant definition.
     */
    private class Helper implements PatternDSLVisitor {

        private boolean isWithoutCall = true;

        @Override
        public void visit(ASTCallExpression node) {
            isWithoutCall = false;
        }

        public boolean testForCalls(ASTConstant node) {
            node.accept(getRealThis());
            return isWithoutCall;
        }
    }


}
