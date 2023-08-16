package de.parallelpatterndsl.patterndsl.coco;


import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTCallExpressionCoCo;

import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTLiteralExpression;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class PredefinedFunctionParameterCountCoCo implements PatternDSLASTCallExpressionCoCo {

    @Override
    public void check(ASTCallExpression node) {
        String name = ((ASTNameExpression)node.getCall()).getName();
        if (PredefinedFunctions.contains(name)) {
            if (node.getArguments().getExpressionList().size() < PredefinedFunctions.minParameters(name, node.get_SourcePositionStart())) {
                Log.error(node.get_SourcePositionStart() + " Function " + name + " needs at least " + PredefinedFunctions.minParameters(name, node.get_SourcePositionStart()) + " parameter(s)!");
            }
            if (node.getArguments().getExpressionList().size() > PredefinedFunctions.maxParameters(name, node.get_SourcePositionStart())) {
                Log.error(node.get_SourcePositionStart() + " Function " + name + " needs at most " + PredefinedFunctions.maxParameters(name, node.get_SourcePositionStart()) + " parameter(s)!");
            }
        }
    }

}
