package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTIndexAccessExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTIndexAccessExpressionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.se_rwth.commons.logging.Log;

import java.util.Optional;

/**
 * CoCo that checks if the access notation ([]) is used on Lists.
 */
public class IndexAccessListExistsCoCo implements PatternDSLASTIndexAccessExpressionCoCo {



    @Override
    public void check(ASTIndexAccessExpression node) {
        ASTExpression exp = node.getExpression();
        if (exp instanceof ASTIndexAccessExpression) {
            return;
        } else if(exp instanceof ASTNameExpression) {
            String name = ((ASTNameExpression) exp).getName();
            Optional<VariableSymbol> symbol = node.getEnclosingScope().resolve(name, VariableSymbol.KIND);
            if (symbol.isPresent()) {
                if (symbol.get().getType() instanceof ASTListType) {
                    return;
                }
            } else {
                Optional<FunctionParameterSymbol> sym = node.getEnclosingScope().resolve(name, FunctionParameterSymbol.KIND);
                if (sym.isPresent()) {
                    if (sym.get().getType() instanceof ASTListType) {
                        return;
                    }
                }
            }
        }
        Log.error(node.get_SourcePositionStart() + " Expression not of List type");
    }
}
