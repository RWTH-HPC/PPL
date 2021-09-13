package de.parallelpatterndsl.patterndsl._symboltable;

import de.monticore.expressions.commonexpressions._ast.ASTCallExpression;
import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.monticore.literals.literals._ast.ASTIntLiteral;
import de.monticore.literals.literals._ast.ASTLiteral;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;

import java.util.ArrayList;
import java.util.Optional;

/**
 * A class which populates the Variable shape, by traversing the AST after handling code inclusion and constant replacements.
 */
public class VariableShapeHandler implements PatternDSLVisitor {

    /**
     * The module on which the visitor shall extend the variable symbols.
     */
    private ASTModule module;

    public VariableShapeHandler(ASTModule module) {
        this.module = module;
    }

    /**
     * Execution of the of the visitor to generate the shape of variables.
     */
    public void generateShapeForVariables() {
        module.accept(this.getRealThis());
    }

    @Override
    public void visit(ASTVariable variable) {
        if (variable.getType() instanceof ASTListType && variable.isPresentExpression()) {
            ASTExpression exp = variable.getExpression();
            ArrayList<Integer> shape = new ArrayList<>();

            // handle instantiation with init function
            if (exp instanceof ASTCallExpression) {
                if (((ASTCallExpression) exp).getExpression() instanceof ASTNameExpression) {
                    if (((ASTNameExpression) ((ASTCallExpression) exp).getExpression()).getName().equals("init_List")) {
                        ASTExpression dimensions = ((ASTCallExpression) exp).getArguments().getExpression(0);
                        if (dimensions instanceof ASTListExpression) {
                            for (ASTExpression value: ((ASTListExpression) dimensions).getExpressionList() ) {
                                if (value instanceof ASTLitExpression) {
                                    ASTLiteral literal = ((ASTLitExpression) value).getLiteral();
                                    if (literal instanceof ASTIntLiteral) {
                                        shape.add(((ASTIntLiteral) literal).getValue());
                                    } else {
                                        shape = new ArrayList<>();
                                        break;
                                    }
                                } else {
                                    shape = new ArrayList<>();
                                    break;
                                }
                            }
                        }
                    }
                }
            }


            // handle instantiation with List expressions
            while (exp instanceof ASTListExpression) {
                Integer length = ((ASTListExpression) exp).sizeExpressions();
                shape.add(length);
                if (length > 0) {
                    exp = ((ASTListExpression) exp).getExpression(0);
                }
            }


            // add shape to variable symbol
            Optional<VariableSymbol> resolve = variable.getEnclosingScope().resolve(variable.getName(), VariableSymbol.KIND);
            if (resolve.isPresent()) {
                resolve.get().setShape(shape);
            }
        }
    }
}
