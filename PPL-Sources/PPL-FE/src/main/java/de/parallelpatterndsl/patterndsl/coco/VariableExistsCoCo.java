package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.commonexpressions._cocos.CommonExpressionsASTNameExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTAssignmentExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTAssignmentExpressionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTIndexAccessExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.ConstantSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a variable, parameter, function or constant is defined.
 */
public class VariableExistsCoCo implements CommonExpressionsASTNameExpressionCoCo {

    @Override
    public void check(ASTNameExpression node) {
        String name = node.getName();
        if (name.startsWith("INDEX") || PredefinedFunctions.contains(name)) {
            return;
        }
        if (node.isPresentEnclosingScope()) {
            if (node.getEnclosingScope().resolveMany(node.getName(), FunctionParameterSymbol.KIND).size() < 1 && node.getEnclosingScope().resolveMany(node.getName(), VariableSymbol.KIND).size() < 1 && node.getEnclosingScope().resolveMany(node.getName(), FunctionSymbol.KIND).size() < 1 && node.getEnclosingScope().resolveMany(node.getName(), ConstantSymbol.KIND).size() < 1) {
                Log.error(node.get_SourcePositionStart() +" Variable/Function not defined: " + node.getName());
            }

        }
    }


}
