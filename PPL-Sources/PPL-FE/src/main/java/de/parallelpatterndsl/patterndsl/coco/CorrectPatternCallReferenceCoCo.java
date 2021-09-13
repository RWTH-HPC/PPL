package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTPatternCallStatementCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class CorrectPatternCallReferenceCoCo implements PatternDSLASTPatternCallStatementCoCo {

    @Override
    public void check(ASTPatternCallStatement node) {
        if(node.isPresentEnclosingScope()) {
            if (((FunctionSymbol) node.getEnclosingScope().resolve(node.getName(), FunctionSymbol.KIND).get()).getPattern().isPresentSerial()) {
                Log.error(node.get_SourcePositionStart() + " No parallel pattern defined for: " + node.getName());
            }
        }
    }

}
