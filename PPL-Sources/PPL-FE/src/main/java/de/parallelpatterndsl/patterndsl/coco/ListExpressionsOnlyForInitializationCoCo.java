package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTListExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._ast.ASTVariable;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTModuleCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLParentAwareVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if constants are only read from.
 */
public class ListExpressionsOnlyForInitializationCoCo implements PatternDSLASTModuleCoCo {

    @Override
    public void check(ASTModule node) {
        Helper helper = new Helper();

        node.accept(helper);
    }




    private class Helper extends PatternDSLParentAwareVisitor {

        @Override
        public void visit(ASTListExpression node) {
            if (getParent().isPresent()) {
                if ( !(getParent().get() instanceof ASTVariable || getParent().get() instanceof ASTPatternCallStatement || getParent().get() instanceof ASTListExpression)) {
                    Log.error(node.get_SourcePositionStart() + " List Expression used outside of a direct assignment!");
                }
            } else {
                Log.error(node.get_SourcePositionStart() + " List Expression used outside of a direct assignment!");
            }
        }
    }
}
