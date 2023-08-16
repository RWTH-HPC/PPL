package de.parallelpatterndsl.patterndsl.coco;


import de.monticore.ast.ASTNode;
import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;

import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTModuleCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTVariableCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.se_rwth.commons.logging.Log;

import java.util.HashSet;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class ListsMustBeInitializedCoCo implements PatternDSLASTModuleCoCo {


    @Override
    public void check(ASTModule node) {
        AllVariablesInitialized tester = new AllVariablesInitialized();
        tester.hasInitClause(node);
    }

    private class AllVariablesInitialized implements PatternDSLVisitor {


        private HashSet<String> present = new HashSet<>();

        private ASTModule module;

        public void hasInitClause(ASTModule node){
            module = node;
            node.accept(this.getRealThis());
            return;
        }

        @Override
        public void visit(ASTVariable node) {
            if (node.getType() instanceof ASTListType) {
                if (node.isPresentExpression()) {
                    if (node.getExpression() instanceof ASTCallExpression) {
                        if (!((ASTNameExpression)((ASTCallExpression) node.getExpression()).getCall()).getName().equals("init_List")) {
                            Log.error(node.get_SourcePositionStart() + "Variable " + node.getName() + " not initialized with init_List or List expression.");
                        } else {
                            present.add(node.getName());
                        }
                    } else if (node.getExpression() instanceof ASTListExpression) {
                        present.add(node.getName());
                    } else {
                        Log.error(node.get_SourcePositionStart() + "Variable " + node.getName() + " must be initialized with init_List() or List expression.");
                    }
                }
            } else {
                present.add(node.getName());
            }
        }

        @Override
        public void visit(ASTAssignmentExpression node) {
            String name ="";
            if (node.getLeft() instanceof ASTNameExpression) {
                name = ((ASTNameExpression) node.getLeft()).getName();
            } else if (node.getLeft() instanceof ASTIndexAccessExpression) {
                ASTNode currentLevel = node.getLeft();

                while (currentLevel instanceof ASTIndexAccessExpression) {
                    currentLevel = ((ASTIndexAccessExpression) currentLevel).getIndexAccess();
                }
                if (currentLevel instanceof ASTNameExpression) {
                    name = ((ASTNameExpression) currentLevel).getName();
                }
            }

            if (!present.contains(name)) {
                if (name.equals("")) {
                    Log.error(node.get_SourcePositionStart() + "Variable not recognized. Possible Syntax error.");
                }
                if (node.getRight() instanceof ASTCallExpression) {
                    if (((ASTNameExpression)((ASTCallExpression) node.getRight()).getCall()).getName().equals("init_List")) {
                        present.add(name);
                    } else {
                        Log.error(node.get_SourcePositionStart() + "Variable " + name + " not initialized with init_List or List expression.");
                    }
                } else if (node.getRight() instanceof ASTListExpression) {
                    present.add(name);
                } else if (node.getEnclosingScope().resolve(name, FunctionParameterSymbol.KIND).isPresent()) {
                    present.add(name);
                } else {
                        Log.error(node.get_SourcePositionStart() + "Variable " + name + " not initialized with init_List or List expression.");
                }
            }
        }

        @Override
        public void visit(ASTFunction node) {
            HashSet<String> newPresent = new HashSet<>();

            HashSet<String> parameters = new HashSet<>();
            node.getFunctionParameters().getFunctionParameterList().stream().map(x -> parameters.add(x.getName()));
            if (node.isPresentFunctionParameter()) {
                parameters.add(node.getFunctionParameter().getName());
            }

            for (String name : parameters ) {
                if (module.getEnclosingScope().resolve(name, VariableSymbol.KIND).isPresent()) {
                    newPresent.add(name);
                }
                if (!node.getPatternType().isPresentSerial()) {
                    if (node.isPresentFunctionParameter()) {
                        newPresent.add(node.getFunctionParameter().getName());
                    } else {
                        Log.error(node.get_SourcePositionStart() + " No return variable for this pattern defined: " + node.getName());
                    }
                }
            }
            present = newPresent;
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
