package de.parallelpatterndsl.patterndsl.coco.Helper;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * A class that implements the visitor pattern, to traverse the AST and check if certain variables are only read from.
 */
public class ReadOnlyHelperGlobalVars implements PatternDSLVisitor {

    private boolean readOnly = true;

    private ArrayList<String> parameters;

    public ReadOnlyHelperGlobalVars(ArrayList<String> parameters) {
        this.parameters = parameters;
    }

    public boolean isReadOnly(ASTModule node) {
        for (ASTDefinition def : node.getDefinitionList().stream().filter(c -> c instanceof ASTFunction).collect(Collectors.toList()) ) {
            ASTFunction func = (ASTFunction) def;
            func.accept(getRealThis());
        }
        return readOnly;
    }

    @Override
    public void traverse(ASTBlockElement node) {
        if (node.isPresentVariable()) {
            return;
        } else if (node.isPresentStatement()) {
            if (node.getStatement() instanceof ASTForStatement) {
                ((ASTForStatement) node.getStatement()).getBlockStatement().accept(getRealThis());
            } else if (node.getStatement() instanceof ASTIfStatement) {
                ((ASTIfStatement) node.getStatement()).getThenStatement().accept(getRealThis());
                if (((ASTIfStatement) node.getStatement()).isPresentElseStatement()) {
                    ((ASTIfStatement) node.getStatement()).getElseStatement().accept(getRealThis());
                }
            } else if ((node.getStatement() instanceof ASTWhileStatement)) {
                ((ASTWhileStatement) node.getStatement()).getBlockStatement().accept(getRealThis());
            }
        } else if (node.isPresentExpression()) {
            ASTExpression exp = node.getExpression();
            if (exp instanceof ASTAssignmentExpression) {
                exp.accept(getRealThis());
            } else if (exp instanceof ASTAssignmentByMultiplyExpression) {
                exp.accept(getRealThis());
            } else if (exp instanceof ASTAssignmentByIncreaseExpression) {
                exp.accept(getRealThis());
            } else if (exp instanceof ASTAssignmentByDecreaseExpression) {
                exp.accept(getRealThis());
            } else if (exp instanceof ASTDecrementExpression) {
                exp.accept(getRealThis());
            } else if (exp instanceof ASTIncrementExpression) {
                exp.accept(getRealThis());
            }
        }
    }

    @Override
    public void traverse(ASTIncrementExpression node) {
        node.getLeft().accept(getRealThis());
    }

    @Override
    public void traverse(ASTAssignmentByDecreaseExpression node) {
        node.getLeft().accept(getRealThis());
    }

    @Override
    public void traverse(ASTIfStatement node) {
        node.getThenStatement().accept(getRealThis());
        if (node.getElseStatementOpt().isPresent()) {
            node.getElseStatement().accept(getRealThis());
        }
    }


    @Override
    public void traverse(ASTAssignmentByIncreaseExpression node) {
        node.getLeft().accept(getRealThis());
    }

    @Override
    public void traverse(ASTAssignmentByMultiplyExpression node) {
        node.getLeft().accept(getRealThis());
    }

    @Override
    public void traverse(ASTAssignmentExpression node) {
        node.getLeft().accept(getRealThis());
    }

    @Override
    public void traverse(ASTIndexAccessExpression node) {node.getExpression().accept(getRealThis());}

    @Override
    public void visit(ASTNameExpression node) {
        for (String element: parameters) {
            if(node.getName().equals(element)) {
                Log.error(node.get_SourcePositionStart() + " Global variable changed: "+ node.getName());
                readOnly = false;
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
