package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTCallExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class NoCopyInFrontendCoCo implements PatternDSLASTCallExpressionCoCo {

    @Override
    public void check(ASTCallExpression node) {
        if(node.isPresentEnclosingScope()) {
            String name = ((ASTNameExpression)node.getCall()).getName();
            if (name.equals("copy")) {
                Log.error(node.get_SourcePositionStart() + " The function copy may not be used! The programming model utilizes a call by value strategy, therefore array data is automatically copied on assignment. Parallel patterns are an exception from this rule, for performance reasons.");
            }
        }
    }

}
