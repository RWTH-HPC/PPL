package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTCallExpression;
import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.commonexpressions._cocos.CommonExpressionsASTCallExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class CorrectCallReferenceCoCo implements CommonExpressionsASTCallExpressionCoCo {

    @Override
    public void check(ASTCallExpression node) {
        if(node.isPresentEnclosingScope()) {
            String name = ((ASTNameExpression)node.getExpression()).getName();
            if (PredefinedFunctions.contains(name)) {
                return;
            }
            if (!((FunctionSymbol) node.getEnclosingScope().resolve(name , FunctionSymbol.KIND).get()).getPattern().isPresentSerial()) {
                Log.error(node.get_SourcePositionStart() + " No function defined for: " + name);
            }
        }
    }

}
