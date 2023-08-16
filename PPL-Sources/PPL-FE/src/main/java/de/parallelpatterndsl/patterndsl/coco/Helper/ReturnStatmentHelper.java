package de.parallelpatterndsl.patterndsl.coco.Helper;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * A class that implements the visitor pattern, to traverse the AST and check if return statements are present if necessary.
 */
public class ReturnStatmentHelper implements PatternDSLVisitor {

    private boolean existsReturn = false;

    public boolean getExistsReturn(ASTFunction node) {
        node.getBlockStatement().accept(getRealThis());
        return existsReturn;
    }

    @Override
    public void visit(ASTReturnStatement node) {
        if (!node.isPresentReturnExpression()) {
            Log.error(node.get_SourcePositionStart() + " Return statement is empty, please provide a value to be returned.");
        }
        existsReturn = true;
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
