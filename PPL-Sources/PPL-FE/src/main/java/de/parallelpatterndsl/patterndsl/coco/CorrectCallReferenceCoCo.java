package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTCallExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class CorrectCallReferenceCoCo implements PatternDSLASTCallExpressionCoCo {

    @Override
    public void check(ASTCallExpression node) {
        if(node.isPresentEnclosingScope()) {
            String name = ((ASTNameExpression)node.getCall()).getName();
            if (PredefinedFunctions.contains(name)) {
                return;
            }
            if (node.getEnclosingScope().resolve(name , FunctionSymbol.KIND).isPresent()) {
                FunctionSymbol symbol = (FunctionSymbol) node.getEnclosingScope().resolve(name , FunctionSymbol.KIND).get();
                if (!symbol.getPattern().isPresentSerial()) {
                    Log.error(node.get_SourcePositionStart() + " Function defined as parallel pattern, not as sequential function: " + name);
                }
                if (node.getArguments().sizeExpressions() != symbol.getParameterCount()) {
                    Log.error(node.get_SourcePositionStart() + " Function call parameter mismatch for: "  + name + " Provided: " + node.getArguments().sizeExpressions() + ". Expected: " + symbol.getParameterCount());
                }
            } else {
                Log.error(node.get_SourcePositionStart() + " No function defined for: " + name);
            }

        }
    }

}
