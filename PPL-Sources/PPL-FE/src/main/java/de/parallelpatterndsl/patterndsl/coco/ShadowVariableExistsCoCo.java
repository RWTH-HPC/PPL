package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._ast.ASTVariable;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTIndexAccessExpressionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTVariableCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.se_rwth.commons.logging.Log;

import java.util.Optional;

/**
 * CoCo that checks if the access notation ([]) is used on Lists.
 */
public class ShadowVariableExistsCoCo implements PatternDSLASTVariableCoCo {



    @Override
    public void check(ASTVariable node) {
        String name = node.getName();
        if (node.getEnclosingScope().resolveMany(name, VariableSymbol.KIND).size() > 1 || node.getEnclosingScope().resolveMany(name, FunctionParameterSymbol.KIND).size() > 1) {
            Log.error(node.get_SourcePositionStart() + " Variable: " + name + " is defined multiple times. Shadowing is not allowed within the PPL.");
        }

    }
}
