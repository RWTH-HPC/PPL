package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTModuleCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLParentAwareVisitor;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
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
                if ( !(getParent().get() instanceof ASTVariable || getParent().get() instanceof ASTListExpression)) {
                    /** Argument added**/
                    System.out.println(getParent().get().toString());
                    System.out.println(getParent().get().toString());
                    Log.error(node.get_SourcePositionStart() + " List Expression used outside of a direct assignment!");


                }
            } else {

                Log.error(node.get_SourcePositionStart() + " List Expression used outside of a direct assignment! (parent not present case)");
            }
        }

        @Override
        public void traverse(ASTCallExpression node) {
            if (node.getCall() instanceof ASTNameExpression) {
                if (!((ASTNameExpression) node.getCall()).getName().equals("init_List")) {
                    node.getCall().accept(getRealThis());
                    node.getArguments().accept(getRealThis());
                }
            }
        }

        @Override
        public void traverse(ASTPatternCallStatement node) {
            node.getLeft().accept(getRealThis());
            node.getArguments().accept(getRealThis());
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
