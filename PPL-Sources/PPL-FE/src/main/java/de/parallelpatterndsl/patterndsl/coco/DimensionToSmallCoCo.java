package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._ast.ASTType;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTIndexAccessExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.ConstantSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

import java.util.Optional;

/**
 * CoCo that checks, if the depth of the accesses is smaller than the dimension of the variable.
 */
public class DimensionToSmallCoCo implements PatternDSLASTIndexAccessExpressionCoCo {

    @Override
    public void check(ASTIndexAccessExpression node) {
        int depth = 0;

        ASTExpression runningExpression = node;
        while (runningExpression instanceof ASTIndexAccessExpression) {
            runningExpression = ((ASTIndexAccessExpression) runningExpression).getIndexAccess();
            depth++;
        }
        if (runningExpression instanceof ASTNameExpression) {
            Optional<VariableSymbol> variableSymbol = node.getEnclosingScope().resolve(((ASTNameExpression) runningExpression).getName(), VariableSymbol.KIND);
            if (variableSymbol.isPresent()) {
                int dimension = 0;
                ASTType type = variableSymbol.get().getType();

                while (type instanceof ASTListType) {
                    type = ((ASTListType) type).getType();
                    dimension++;
                }
                if (dimension < depth) {
                    Log.error(node.get_SourcePositionStart() + "Dimension mismatch for variable: " + ((ASTNameExpression) runningExpression).getName() + ". access depth: " + depth + ". possible depth: " + dimension);
                }
            } else {
                Optional<FunctionParameterSymbol> functionParameterSymbol = node.getEnclosingScope().resolve(((ASTNameExpression) runningExpression).getName(), FunctionParameterSymbol.KIND);
                if (functionParameterSymbol.isPresent()) {
                    int dimension = 0;
                    ASTType type = functionParameterSymbol.get().getType();

                    while (type instanceof ASTListType) {
                        type = ((ASTListType) type).getType();
                        dimension++;
                    }

                    if (dimension < depth) {
                        Log.error(node.get_SourcePositionStart() + "Dimension mismatch for variable: " + ((ASTNameExpression) runningExpression).getName() + ". access depth: " + depth + ". possible depth: " + dimension);
                    }
                }
            }
        }
    }


}
